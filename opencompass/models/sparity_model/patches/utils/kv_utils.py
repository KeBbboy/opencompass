class DynamicCacheSplitHeadFlatten(Cache):
    """adapt from https://github.com/FFY0/AdaKV."""

    def __init__(self) -> None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),
                   tuple(self.value_cache[layer_idx]))

    def __getitem__(
            self,
            layer_idx: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),
                    tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(
                f'Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}'
            )

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs['head_lens']
            cu_klen = cache_kwargs['cu_klen']

            import nvtx
            copy_old_rng = nvtx.start_range('copy old')
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(
                self.key_cache[layer_idx].view(-1, dim),
                key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(
                self.value_cache[layer_idx].view(-1, dim),
                value_states.view(-1, dim), head_lens, cu_klen)

            nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(
            self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the
        legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx],
                              self.value_cache[layer_idx]), )
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> 'DynamicCacheEachHead':
        """Converts a cache in the legacy cache format into an equivalent
        `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
