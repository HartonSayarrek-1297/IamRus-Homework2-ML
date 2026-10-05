import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import SGDRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

np.random.seed(67) # кстати сид 42 выдал результаты чуть-чуть похуже

dataset = load_dataset("gvlassis/california_housing")
df = dataset['train'].to_pandas()

print("Форма датасета: ", df.shape)
print("Колонки: ", list(df.columns))

feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                 'Population', 'AveOccup', 'Latitude', 'Longitude']
target_name = 'MedHouseVal'


print("ПРИЗНАКИ (что знаем о районе):")
for i, name in enumerate(feature_names, 1):
    print(f"  {i}. {name}")

print(f"\nЦЕЛЕВАЯ ПЕРЕМЕННАЯ (что предсказываем): {target_name}")
print(f"  Единицы: 100 000 $  (2.5 = 250 000 $)")


print("\nПЕРВЫЕ 5 ОБЪЕКТОВ (как таблица)")
sample = df.head(5).copy()

sample.index = range(1, 6)
print(sample.round(3))

X = df[feature_names].values
y = df[target_name].values

print("X:", X.shape, "y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape} ({len(X_train) / len(X) * 100:.0f}%)")
print(f"Test: {X_test.shape} ({len(X_test) / len(X) * 100:.0f}%)")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_features = X_train.shape[1]

print("\nПЕРВЫЕ 5 ОБЪЕКТОВ ПОСЛЕ МАСШТАБИРОВАНИЯ (как нампай-массив)")
headmaster = X_train[0:4].copy()
headcrusher = X_test[0:4].copy()

print("Обучающая выборка\n",headmaster.round(3) )
print("Тестовая выборка\n", headcrusher.round(3) )

# model = Sequential([Dense(1, input_shape=(n_features,), activation='linear')])

# model.summary()

# model.compile(
    # optimizer=Adam(learning_rate=0.01),
    # loss='mse',
    # metrics=['mae']
# )

# history = model.fit(
    # X_train, y_train,
    # epochs=50,
    # batch_size=32,
    # validation_split=0.2
# )

# print("Обучение завершено!")

# fig, ax = plt.subplots(figsize=(9, 5))

# ax.plot(history.history['loss'], label='train loss', linewidth=2)
# ax.plot(history.history['val_loss'], label='val loss', linewidth=2)
# ax.set_xlabel("Эпоха")
# ax.set_ylabel("MSE")
# ax.set_title("Кривые обучения")
# ax.legend()
# ax.set_yscale('log')
# plt.show()



# Ручная, рабоче-крестьянская реализация линейной регрессии (ура, товарищи)

def linear_regression_manual_step(x_list, y_target, w, b, lr): # Функция считает регрессию для 1 экземляра (8 признаков + целевая переменная)
    y_pred = x_list @ w + b
    loss = np.mean((y_pred - y_target)**2)
    
    dw = -2 * np.mean(x_list * (y_target - y_pred))
    db = -2 * np.mean(y_target - y_pred)

    w -= dw * lr
    b -= db * lr
        
    return w, b, loss

def linregman_global(train_data, targets, epochs=100, type='Minibatch32'): # А тут скейлим персональную регрессию для нескольких экземпляров и прогоняем несколько эпох
    
    lr = 0.01
    data_count = 32 if type=='Minibatch32' else train_data.shape[0]
    w, b = np.random.normal(1, size=train_data.shape[1] ), np.random.randn(1)
    
    loss_history = [] # Будет двумерный массив
    indices = list(np.random.choice(train_data.shape[0], size=data_count, replace=False)) # генерим набор индексов для обучения по заветам mini-batch-gradient-descent
    for ep in range(epochs):
        example_loss_current = [] # Это набор персональных функций потерь для нескольких экземпляров - одномерный массив
                
        for i in indices:
            w, b, loss = linear_regression_manual_step(train_data[i], targets[i], w, b, lr)
            example_loss_current.append(loss) # Добавляем персональную функцию потерь (для одного экземпляра)
        loss_history.append(example_loss_current)
    
    return list(map(list, zip(*loss_history))), w, b

hist, w_res, b_res = linregman_global(X_train, y_train)
hist1 = hist[0]
hist2 = hist[1]
hist3 = hist[15]

plt.figure(figsize=(9, 5))
plt.plot(hist1, color='blue', linewidth=2)
plt.plot(hist2, color='red', linewidth=2)
plt.plot(hist3, color='green', linewidth=2)
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.title("Ручной GD: кривая обучения")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()



# Время тестов (я ща жоска её буду мариновать, я серьёзно, это вам не фури-пицца)

y_pred_test = X_test @ w_res + b_res

print("Первые 10 предсказаний vs реальных значений:")
comparison = pd.DataFrame({
    'Реальная цена': y_test[:10].round(3),
    'Предсказание':  y_pred_test[:10].round(3),
    'Ошибка':        (y_pred_test[:10] - y_test[:10]).round(3)
})
print(comparison)