from feast import Entity, Field, FeatureView, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

# 1️⃣ Define the source — your processed CSV data
stock_source = FileSource(
    path="../processed_data/processed_v0/",       # path relative to feature_repo/
    event_timestamp_column="timestamp",
)

# 2️⃣ Define the entity
stock = Entity(name="stock_symbol", join_keys=["stock_symbol"])

# 3️⃣ Define the feature view
stock_features_view = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
        Field(name="target", dtype=Int64),
    ],
    online=True,
    source=stock_source,
)
