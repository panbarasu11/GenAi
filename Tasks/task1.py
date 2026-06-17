import matplotlib.pyplot as plt


hours_studied = [1, 2, 3, 5, 6, 8]
actual_marks = [10, 19, 28, 38, 51, 60]


weights = range(1, 21)
mse_list = []

best_weight = None
lowest_mse = float("inf")

for weight in weights:
    predicted = [weight * h for h in hours_studied]

    mse = sum((p - a) ** 2 for p, a in zip(predicted, actual_marks))
    mse /= len(actual_marks)

    mse_list.append(mse)

    if mse < lowest_mse:
        lowest_mse = mse
        best_weight = weight

print("Best Weight:", best_weight)
print("Lowest MSE:", lowest_mse)

plt.plot(list(weights), mse_list, marker='o')
plt.xlabel("Weight")
plt.ylabel("MSE")
plt.title("Weight vs MSE")
plt.grid(True)
plt.show()

