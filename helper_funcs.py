def preprocess_data(df):
  '''
  This function will:
  1. Turn the 'saledate' feature into more meaningfull datetime features.
  2. Drop the original 'saledate' column.
  3. Fill missing values in numeric features
  4. Turn cateogorical features into numerical values and fill the missing values.
  5. Return the trnasformed df
  '''

  df["saleYear"] = df.saledate.dt.year
  df["saleMonth"] = df.saledate.dt.month
  df["saleDay"] = df.saledate.dt.day
  df["saleDayOfWeek"] = df.saledate.dt.dayofweek
  df["saleDayOfYear"] = df.saledate.dt.dayofyear

  df.drop("saledate", axis=1, inplace=True)

  # Fill the numeric rows with median
  for label, content in df.items():
      if pd.api.types.is_numeric_dtype(content):
          if pd.isnull(content).sum():
              # Add a binary column which tells us if the data was missing or not
              df[label+"_is_missing"] = pd.isnull(content)
              # Fill missing numeric values with median
              df[label] = content.fillna(content.median())

      # turn cateogircal variables into numbers and fill missing
      if not pd.api.types.is_numeric_dtype(content):
          df[label+"_is_missing"] = pd.isnull(content)
          # We add +1 to the category code because pandas encodes missing categories as -1
          df[label] = pd.Categorical(content).codes+1

  return df


def rmsle(y_test, y_preds):
  '''
  calculates root mean squared log error between predicitons and true labels
  '''
  return np.sqrt(mean_squared_log_error(y_test, y_preds))


def get_scores(model):
  '''
  returns MAE, RMSLE, R^2 for given model on the predefined X_train and X_valid data splits
  '''
  train_preds = model.predict(X_train)
  val_preds = model.predict(X_valid)
  scores = {'Training MAE': mean_absolute_error(y_train, train_preds),
            'Valid MAE': mean_absolute_error(y_valid, val_preds),
            'Training RMSLE': rmsle(y_train, train_preds),
            'Valid RMSLE': rmsle(y_valid, val_preds),
            'Training R^2': r2_score(y_train, train_preds),
            'Valid R^2': r2_score(y_valid, val_preds)}
  return scores


def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({'features': columns,
                      'feature importance': importances})
          .sort_values('feature importance', ascending=False)
          .reset_index(drop=True))

    # plot df
    sns.barplot(x="feature importance",
                y="features",
                data=df[:n],
                orient="h")
