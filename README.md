# SwiftAnalytics

**SwiftAnalytics** — це нативна, високопродуктивна, модульна екосистема для аналізу даних та машинного навчання на Swift, оптимізована для Apple Silicon (M-series), архітектури Unified Memory (UMA) та суворої багатопотоковості Swift 6 (Strict Concurrency).

Бібліотека поєднує в собі переваги апаратно-прискорених тензорних обчислень через **MLX** на GPU та оптимізованих SIMD-підпрограм **Accelerate (vDSP / LAPACK)** на CPU для максимальної швидкодії на Apple пристроях.

---

## 🚀 Основні Модулі

* **`SwiftDataFrame`** — швидкі колоночні маніпуляції даними з нульовим копіюванням у пам'яті (zero-copy) на базі `Apache Arrow`.
* **`SwiftStats`** — векторизовані статистичні тести, розподіли та описова статистика за допомогою `Accelerate vDSP`.
* **`SwiftPreprocessing`** — масштабування, кодування категорій, дискретизація та побудова конвеєрів (`Pipeline`).
* **`SwiftML`** — класичні моделі навчання (лінійна/логістична регресія, дерева ухвалення рішень, Random Forest, GBDT).
* **`SwiftCluster`** — кластеризація (K-Means) та зниження розмірності (PCA, DBSCAN).
* **`SwiftOptimize`** — перехресна перевірка (`KFold`) та паралельний пошук гіперпараметрів (`GridSearchCV`).
* **`SwiftForecast`** — аналіз часових рядів (адитивна/мультиплікативна декомпозиція, Holt-Winters, ARIMA, фільтр Калмана).
* **`SwiftNLP`** — токенізація (Word / subword BPE токенізатор) та статичні ембеддінги.
* **`SwiftExplain`** — локальна інтерпретація моделей чорної скриньки за допомогою паралельного алгоритму `KernelSHAP`.
* **`SwiftLLM`** — нативна генерація тексту на GPU за допомогою казуальних трансформер-декодерів та MLX.
* **`SwiftPrivacy`** — криптографічне машинне навчання на зашифрованих даних (Ring-LWE, PNNS) на базі гомоморфного шифрування.

---

## 📊 Порівняння Продуктивності (Swift vs Python)

Нижче наведені медіанні часові результати бенчмарків виконання на архітектурі **Apple Silicon M-series (macOS 15 / arm64)** порівняно з популярними Python-бібліотеками (Scikit-Learn, NumPy, SHAP, Statsmodels, PyTorch):

| Тест (Benchmark) | Swift (ms) | Python (ms) | Прискорення | Переможець |
| :--- | :---: | :---: | :---: | :---: |
| **Pearson Correlation** (500k elements) | 0.777 ms | 1.220 ms | 1.57x | 🟢 Swift |
| **Random Forest fit** (1k samples x 4 features) | 4.630 ms | 23.626 ms | 5.10x | 🟢 Swift |
| **Holt-Winters fit** (50k points, period=12) | 6.349 ms | 135.124 ms | 21.28x | 🟢 Swift |
| **ARIMA fit** (50k points) | 2.297 ms | 204.394 ms | 89.00x | 🟢 Swift |
| **Kalman Filter 1D** (10k observations) | 44.709 ms | 81.172 ms | 1.82x | 🟢 Swift |
| **KernelSHAP Explain** (5 features, 100 coalitions) | 0.172 ms | 0.464 ms | 2.69x | 🟢 Swift |
| **RingLWE Encrypt/Decrypt** (vector size=64) | 0.020 ms | 0.286 ms | 14.28x | 🟢 Swift |
| **PNNS Classify** (50 DB vectors, size=64) | 0.231 ms | 2.903 ms | 12.54x | 🟢 Swift |
| **LLM Generate** (10 tokens streaming) | 4.924 ms | 3.656 ms | 0.74x | 🔴 Python |
| **LLM Forward Pass** (seqLen=64) | 0.505 ms | 0.513 ms | 1.02x | 🟢 Swift |

---

## 🛠 Архітектурні Рішення та Оптимізація

### 1. Перехід від ООП до DOD (Data-Oriented Design)
Для дерев рішень (`DecisionTree`, `RandomForest` та `GBDT`) класичний об'єктно-орієнтований підхід (де кожен вузол є окремим екземпляром класу з лінками) викликав суттєві накладні витрати на підрахунок посилань (ARC) та призводив до фрагментації кешу процесора (cache misses).
Ми перейшли до DOD-архітектури, представивши дерево у вигляді пласкаго масиву структур `FlatTreeNode`:
* Усі вузли зберігаються послідовно у неперервному блоці пам'яті.
* Навігація лівим/правим нащадками здійснюється за індексами у масиві.
* Це дозволило збільшити швидкість тренування та прогнозування ансамблів у кілька разів завдяки кращій локалізації даних у L1/L2 кешах процесора.

### 2. Апаратна Маршрутизація (Hardware Routing)
Впроваджено гнучку систему вибору обчислювального пристрою (`requestedDevice` / `resolvedDevice`):
* Алгоритми з високим розходженням гілок (branch divergence), такі як дерева рішень, Random Forest чи просторовий пошук (DBSCAN), виконуються суто на CPU.
* Матричні операції (лінійна регресія, K-Means) використовують переваги Metal-ядер Apple Silicon GPU через ліниві обчислення `MLXArray`.

### 3. Опціональна перевірка NaN для SIMD
Для Descriptive Statistics (`mean`, `variance`, `standardDeviation`) ми виявили, що перевірка на наявність `NaN` у масиві створює CPU-пляшку, оскільки звичайна ітерація в Swift займає більше часу, ніж SIMD-обчислення у `vDSP`. У версії 1.0 додано параметр `checkNaN: Bool = true`, який можна вимкнути для критичних за продуктивністю ділянок коду (в бенчмарках встановлено `false`).

---

## ⚠️ Виявлені Вузькі Місця та Недоліки

1. **Тимчасово вища складність I/O для CSV-парсингу**: Наразі реалізація парсингу в `SwiftDataFrame` має вищу тимчасову складність $O(N)$ для операцій введення/виведення, ніж Pandas, через відсутність повноцінного потокового (streaming) парсингу великих файлів. Оптимізація потокового парсингу запланована на наступні етапи розвитку бібліотеки.
2. **Низькорівневий міст з MLX**: `MLXArray` за замовчуванням не реалізує протокол `Sendable`. Це було вирішено за допомогою суворої ізоляції акторів та передачі токенів володіння `WiredMemoryTicket`.

---

## 💻 Швидкий Старт

```swift
import SwiftDataFrame
import SwiftStats
import SwiftML

// 1. Завантаження даних
let df = try DataFrame.readCSV(contentsOf: csvURL)

// 2. Статистичний опис
let summary = try Stats.describe(df["target"].toDoubles()!)
print(summary)

// 3. Навчання моделі
let regressor = LinearRegression(device: .gpu)
try await regressor.fit(features: X, targets: y)
let predictions = try await regressor.predict(features: X_test)
```

---

## 📜 Ліцензія
Проєкт розповсюджується під ліцензією MIT.
