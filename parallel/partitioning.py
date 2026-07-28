from jax.sharding import PartitionSpec as P

class LogicalRules:
    Activation = P(("dp", "fsdp"), None)
    
    # Native mapping of JAX/Megatron FSDP+TP logic inside XLA.
    # Note: Weight shapes in Equinox are stored transposed as [out_features, in_features]
    ColumnLinear = P("tp", "fsdp")
    RowLinear = P("fsdp", "tp")
    Vocab = P("tp", "fsdp")
    Norm = P("fsdp")