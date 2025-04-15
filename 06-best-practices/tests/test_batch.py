import pandas as pd
from datetime import datetime
from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    df = pd.DataFrame(data, columns=columns)

    categorical = ["PULocationID", "DOLocationID"]
    df_actual = prepare_data(df, categorical)

    df_expected = pd.DataFrame(
        data=[
            ("-1", "-1", 9.0),
            ("1", "1", 8.0)
        ],
        columns=["PULocationID", "DOLocationID", "duration"],
    )

    correct_mask = (
        (df_actual["PULocationID"] == df_expected["PULocationID"]) &
        (df_actual["DOLocationID"] == df_expected["DOLocationID"]) &
        (df_actual["duration"].round(2) == df_expected["duration"].round(2))
    )

    print("==== CORRECT ROWS ====")
    print(df_actual[correct_mask])

    assert correct_mask.all()



# import pandas as pd
# from datetime import datetime

# from batch import prepare_data


# def dt(hour, minute, second=0):
#     return datetime(2023, 1, 1, hour, minute, second)


# def test_prepare_data():

#     data = [
#         (None, None, dt(1, 1), dt(1, 10)),
#         (1, 1, dt(1, 2), dt(1, 10)),
#         (1, None, dt(1, 2, 0), dt(1, 2, 59)),
#         (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
#     ]

#     columns = [
#         "PULocationID",
#         "DOLocationID",
#         "tpep_pickup_datetime",
#         "tpep_dropoff_datetime",
#     ]
#     df = pd.DataFrame(data, columns=columns)

#     categorical = ["PULocationID", "DOLocationID"]

#     df_actual = prepare_data(df, categorical)

#     df_expected = pd.DataFrame(
#         data=[
#             ("-1", "-1", 9.0),
#             ("1", "1", 8.0)
#         ],
#         columns=["PULocationID", "DOLocationID", "duration"],
#     )

#     print(df_actual)

#     assert (df_actual["PULocationID"] == df_expected["PULocationID"]).all()
#     assert (df_actual["DOLocationID"] == df_expected["DOLocationID"]).all()
#     assert (df_actual["duration"] - df_expected["duration"]).abs().sum() < 0.0000001    

