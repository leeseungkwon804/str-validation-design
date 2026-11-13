# Data dictionary

## The 50 predictors

All variables come from the published snapshot files, except the `len_*` and `n_*` counts, which are character and item counts derived from the free-text and list fields of the same file, and the `days_*` variables, which are day counts from the snapshot date.

No variable identifying a place (city, country, region, latitude, longitude, neighbourhood) and no listing or host identifier enters the design matrix. Those are held as metadata and used only to construct the splits and to aggregate results. The raw character length of the licence field is excluded as outcome-derived.

| Variable | Group | Type |
|---|---|---|
| `accommodates` | Property type | numeric |
| `bathrooms` | Property type | numeric |
| `bedrooms` | Property type | numeric |
| `beds` | Property type | numeric |
| `minimum_nights` | Minimum and maximum stay | numeric |
| `maximum_nights` | Minimum and maximum stay | numeric |
| `availability_30` | Availability | numeric |
| `availability_60` | Availability | numeric |
| `availability_90` | Availability | numeric |
| `availability_365` | Availability | numeric |
| `number_of_reviews` | Reviews | numeric |
| `number_of_reviews_ltm` | Reviews | numeric |
| `number_of_reviews_l30d` | Reviews | numeric |
| `review_scores_rating` | Reviews | numeric |
| `review_scores_accuracy` | Reviews | numeric |
| `review_scores_cleanliness` | Reviews | numeric |
| `review_scores_checkin` | Reviews | numeric |
| `review_scores_communication` | Reviews | numeric |
| `review_scores_location` | Reviews | numeric |
| `review_scores_value` | Reviews | numeric |
| `reviews_per_month` | Reviews | numeric |
| `host_listings_count` | Host portfolio size | numeric |
| `host_total_listings_count` | Host portfolio size | numeric |
| `calculated_host_listings_count` | Host portfolio size | numeric |
| `calculated_host_listings_count_entire_homes` | Host portfolio size | numeric |
| `calculated_host_listings_count_private_rooms` | Host portfolio size | numeric |
| `calculated_host_listings_count_shared_rooms` | Host portfolio size | numeric |
| `estimated_occupancy_l365d` | Availability | numeric |
| `estimated_revenue_l365d` | Availability | numeric |
| `host_response_rate` | Host conduct | numeric |
| `host_acceptance_rate` | Host conduct | numeric |
| `host_is_superhost` | Host conduct | numeric |
| `host_has_profile_pic` | Host conduct | numeric |
| `host_identity_verified` | Host conduct | numeric |
| `instant_bookable` | Host conduct | numeric |
| `has_availability` | Availability | numeric |
| `property_type` | Property type | categorical (integer-encoded) |
| `room_type` | Property type | categorical (integer-encoded) |
| `host_response_time` | Host conduct | categorical (integer-encoded) |
| `price` | Price | numeric |
| `len_name` | Text length | numeric |
| `len_description` | Text length | numeric |
| `len_neighborhood_overview` | Text length | numeric |
| `len_host_about` | Text length | numeric |
| `len_amenities` | Text length | numeric |
| `n_amenities` | Text length | numeric |
| `n_host_verif` | Host conduct | numeric |
| `days_host_since` | Tenure | numeric |
| `days_first_review` | Tenure | numeric |
| `days_last_review` | Tenure | numeric |

## Names used in `derived/`

Design and model names are byte-identical to those in the manuscript and the supplementary tables.

| Field | Values |
|---|---|
| `design` | 1 Random (conventional); 2 Market hold-out; 3 Temporal (pre to post); 4 Host hold-out; 5 Market by time |
| `model` | Logistic regression; Random forest; Gradient boosting; TabPFN, each optionally suffixed with `, full fold` or `, 10,000 rows` |
| `outcome` | Loose (primary); Strict |
| `group` | Price; Minimum and maximum stay; Property type; Availability; Reviews; Host portfolio size; Host conduct; Text length; Tenure |

