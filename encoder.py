import pandas as pd
class Encoder:
    @staticmethod
    def encode_data(x_train, numerical_columns, d, enc_create=True):
        print "Numerical columns", str(numerical_columns)
        if numerical_columns is not None:
            cat_cols = x_train.drop(numerical_columns, axis=1).fillna('NA')
        else:
            cat_cols = x_train

        label_enc_map = {}

        if enc_create:
            enc = ()
            cat_data = cat_cols.apply(lambda x: d[x.name].fit_transform(x))
        else:
            cat_data = cat_cols.apply(lambda x: d[x.name].transform(x))

        if numerical_columns is not None:
            df = pd.concat([x_train[numerical_columns].fillna(0), cat_data], axis=1)
        else:
            df = cat_data
        return df.values