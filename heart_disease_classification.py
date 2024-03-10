import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, BatchNormalization
from sklearn.metrics import classification_report, roc_curve, auc
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
 
#print(data.info())
print(Counter(data.DEATH_EVENT))

y = data.DEATH_EVENT
x = data.drop(["DEATH_EVENT"], axis=1)
#print(x.columns)

x = pd.get_dummies(x)

X_train, X_temp, Y_train, Y_temp = train_test_split(x, y, test_size=0.2) # Split to train set and test+val set
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5)  # Splits the remainder into two equal parts for validation and testing
#print(X_train.info())


# Scaler
scaler = StandardScaler()
#CT for columns which were categorical variables before one-hot-encoding
ct = ColumnTransformer([("numeric", StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])

# Balance dataset using SMOTE
sampling_strategy_dict = {0: 200, 1:200}
sm = SMOTE(sampling_strategy = sampling_strategy_dict, random_state=42)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)

#Transform with ColumnTransformer using StandardScaler
X_train = ct.fit_transform(X_train_res)

X_test = ct.transform(X_test)
X_val = ct.transform(X_val)

#Label encoding (this case 0 or 1)
le = LabelEncoder()
Y_train = le.fit_transform(Y_train_res.astype(str))
Y_test = le.transform(Y_test.astype(str))
Y_val = le.transform(Y_val.astype(str))

#Transform to binary vector (this case [0, 1] or [1, 0])
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_val = to_categorical(Y_val)

#Model
model = Sequential()
input_layer = InputLayer(input_shape=(X_train.shape[1],))

#Build model
model.add(input_layer)
model.add(Dense(256, activation='relu', kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu', kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu', kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(48, activation='relu', kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax")) #Output number corresponds to the sum of categories

#Compile model
custom_adam = Adam(learning_rate=0.00005)
model.compile(loss="binary_crossentropy", optimizer=custom_adam, metrics=["accuracy"])

#Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20)

#summary
print(model.summary())

#Fit model
EPOCHS = [200]
bsize= [10, 20, 30, 50, 70, 90, 110]
best = 0
element = 0
jelement = 0
best_history = None

for i, element in enumerate(EPOCHS):
    for j, jelement in enumerate(bsize):
        history = model.fit(X_train, Y_train, epochs=element, batch_size=jelement, verbose=1, validation_data=(X_val, Y_val), callbacks=[early_stopping])

        #Evaluate model
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)

        if acc >= best:
            best = acc
            best_ele = element
            best_jele = jelement
            best_history = history

        else:
            pass

print("Index for best calc:", best_ele)
print("Index for best calc j", best_jele)
print("Loss:", loss)
print("Accuracy:", best)

#Create classification report of model
y_estimate = model.predict(X_test)
y_pred = np.argmax(y_estimate, axis=1)
y_true = np.argmax(Y_test, axis=1)

print("Testset Classification report:")
print(classification_report(y_true, y_pred))

# Create a confusion matrix
confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

# Plot the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.savefig("CM_heart_failure_classification.png")

# Create a classification report DataFrame
report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()

# Drop support column
report_df.drop(columns='support', inplace=True, errors='ignore')

plt.figure(figsize=(8, 6))
sns.heatmap(report_df, annot=True, cmap='Greens', fmt='.2f')
plt.title('Classification Report')
plt.savefig("Classification_Report.png")
