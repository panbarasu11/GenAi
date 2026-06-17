import matplotlib.pyplot as plt


hours_studied = [1, 2, 3, 5, 6, 8]
actual_marks = [10, 19, 28, 38, 51, 60]

new_students = [4, 7, 9]

best_w = None
best_b = None
lowest_mse = float("inf")

for weight in range(1, 21):
    for bias in range(-10, 11):

        predicted = [weight * h + bias for h in hours_studied]

        mse = sum((p - a) ** 2 for p, a in zip(predicted, actual_marks))
        mse /= len(actual_marks)

        if mse < lowest_mse:
            lowest_mse = mse
            best_w = weight
            best_b = bias

print("\nBest Weight:", best_w)
print("Best Bias:", best_b)
print("Lowest MSE:", lowest_mse)

# Predictions
for hrs in new_students:
    prediction = best_w * hrs + best_b
    print(f"Student studied {hrs} hrs → predicted marks: {prediction}")