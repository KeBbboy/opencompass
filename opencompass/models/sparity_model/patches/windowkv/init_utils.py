import math
import torch
import torch.nn.functional as F




class WindowKVCluster():
    cached_selected_window_indices = None
    layers_budget = None

    def __init__(self, num_hidden_layers, layer_idx, review_window_size, win_max_attn_scores, shared_layers, suffix_size, \
                beta, max_capacity_prompt, layer_decay_strategy, window_select_strategy):
        
        self.num_hidden_layers = num_hidden_layers
        self.layer_idx = layer_idx
        self.review_window_size = review_window_size
        self.win_max_attn_scores = win_max_attn_scores
        self.shared_layers = shared_layers
        self.suffix_size = suffix_size
        self.beta = beta
        self.max_capacity_prompt = max_capacity_prompt
        self.layer_decay_strategy = layer_decay_strategy
        self.window_select_strategy = window_select_strategy

        

        if WindowKVCluster.layers_budget is None:
            layers_budget = []
            if self.layer_decay_strategy == "arithmetic":
                groups_budget = []
                num_groups = self.num_hidden_layers // self.shared_layers
                last_group_budget = self.max_capacity_prompt * self.shared_layers // self.beta
                first_group_budget = self.max_capacity_prompt * 2 - last_group_budget
                groups_budget.append(first_group_budget)
                steps = (first_group_budget - last_group_budget) // (num_groups - 1)
                for group_idx in range(1, num_groups - 1):
                    cur_group_budget = first_group_budget - group_idx * steps
                    groups_budget.append(cur_group_budget)
                groups_budget.append(last_group_budget)

                for idx in range(num_groups):
                    layers_budget.extend([groups_budget[idx] for _ in range(self.shared_layers)])
            
            WindowKVCluster.layers_budget = layers_budget


        

    def get_window_score(self, window, strategy):
        if strategy == "average":
            window_score = window.mean().item()
        elif strategy == "max":
            values, indices = torch.topk(window, min(self.win_max_attn_scores, window.shape[-1]))
            window_score = values.mean().item()
        return window_score




    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefilling phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        len_review = q_len-self.suffix_size
        

        if len_review <= WindowKVCluster.layers_budget[self.layer_idx]:
            return key_states, value_states


        elif self.layer_idx % self.shared_layers != 0:
            review_key_states = key_states[..., :-self.suffix_size, :]
            review_value_states = value_states[..., :-self.suffix_size, :]
            selected_key_states = []
            selected_value_states = []

            for _i in WindowKVCluster.cached_selected_window_indices:
                start_idx = _i * self.review_window_size
                end_idx = min(start_idx + self.review_window_size, len_review)
                selected_key_states.append(review_key_states[..., start_idx:end_idx, :])
                selected_value_states.append(review_value_states[..., start_idx:end_idx, :])

            k_past_compressed = torch.cat(selected_key_states, dim=-2)
            v_past_compressed = torch.cat(selected_value_states, dim=-2)

            
            k_cur = key_states[..., -self.suffix_size:, :]
            v_cur = value_states[..., -self.suffix_size:, :]

            key_states = torch.cat([k_past_compressed, k_cur], dim = -2)
            value_states = torch.cat([v_past_compressed, v_cur], dim = -2)

            return key_states, value_states
        
        else:

            attn_weights = torch.matmul(query_states[..., -self.suffix_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.suffix_size, self.suffix_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.suffix_size:, -self.suffix_size:] += attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
            attn_weights_sum = attn_weights[:, :, -self.suffix_size:, : -self.suffix_size].sum(dim = -2) 
            

            num_windows = (len_review + self.review_window_size - 1) // self.review_window_size
            review_windows = []
            for _i in range(num_windows):
                start_idx = _i * self.review_window_size
                end_idx = min(start_idx + self.review_window_size, len_review)
                window = attn_weights_sum[:, :, start_idx:end_idx]
                window_score = self.get_window_score(window=window, strategy=self.window_select_strategy)
                review_windows.append((_i, window_score))

            self.bottom_k = math.ceil((len_review - WindowKVCluster.layers_budget[self.layer_idx]) / self.review_window_size)
            
            if self.bottom_k * self.review_window_size >= len_review:
                key_states = key_states[..., -self.suffix_size:, :]
                value_states = value_states[..., -self.suffix_size:, :]

                return key_states, value_states


            else: 
                review_windows = sorted(review_windows, key=lambda x: x[1], reverse=True)[:-self.bottom_k]
                selected_windows_indices = [x[0] for x in review_windows]

                review_key_states = key_states[..., :-self.suffix_size, :]
                review_value_states = value_states[..., :-self.suffix_size, :]
                selected_key_states = []
                selected_value_states = []

                for _i in selected_windows_indices:
                    start_idx = _i * self.review_window_size
                    end_idx = min(start_idx + self.review_window_size, len_review)
                    selected_key_states.append(review_key_states[..., start_idx:end_idx, :])
                    selected_value_states.append(review_value_states[..., start_idx:end_idx, :])


                k_past_compressed = torch.cat(selected_key_states, dim=-2)
                v_past_compressed = torch.cat(selected_value_states, dim=-2)
                
                k_cur = key_states[..., -self.suffix_size:, :]
                v_cur = value_states[..., -self.suffix_size:, :]


                key_states = torch.cat([k_past_compressed, k_cur], dim = -2)
                value_states = torch.cat([v_past_compressed, v_cur], dim = -2)
                WindowKVCluster.cached_selected_window_indices = selected_windows_indices
                return key_states, value_states
            




def init_WindowKV(self, num_hidden_layers):


    if self.config.window_select_strategy == "max":
        win_max_attn_scores = 8 // 2
        suffix_size = 32
    elif self.config.window_select_strategy == "average":
        win_max_attn_scores = 0
        suffix_size = 16
    else:
        raise ValueError(f"Unknown window_select_strategy in qwen: {self.config.window_select_strategy}.")
        


    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = WindowKVCluster(
              # ----- 模型结构参数 -----
              num_hidden_layers = num_hidden_layers,  # 模型总层数（如 Qwen2-7B 有28层）
              layer_idx = self.layer_idx,  # 当前层的索引（0 到 num_hidden_layers-1）

              # ----- 窗口划分参数 -----
              review_window_size = 8,  # 每个窗口包含8个连续token（语义单元）
              suffix_size = suffix_size,  # 后缀窗口大小（始终保留的最后N个token）

              # ----- 窗口选择参数 -----
              win_max_attn_scores = win_max_attn_scores,  # 计算窗口得分时的 top-k 值
              window_select_strategy = self.config.window_select_strategy,  # "max" 或 "average"

              # ----- 层间共享参数 -----
              shared_layers = 8,  # 每8层为一组，组内共享窗口选择索引（减少计算）

              # ----- 预算分配参数 -----
              max_capacity_prompt = self.config.max_capacity_prompt,  # 所有层的平均KV cache大小
              beta = 14,  # 预算衰减系数（控制最后一组的budget比例）
              layer_decay_strategy = "arithmetic",  # 层间预算分配策略：算术递减
        )



