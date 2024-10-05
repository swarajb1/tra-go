import numpy as np

# true random simulation

# reward = 1 percent per day
# risk_reward_ratio = 0.5
# success_rate = 0.5 and rest 0.5 is stop loss hit


for _ in range(15):
    # Create an array of size 10 with random values between -1 and 1
    z = np.random.random(250)

    num_negative = int(0.5 * len(z))
    negative_indexes = np.random.choice(len(z), num_negative, replace=False)

    # Multiply the selected indexes by -1
    z[negative_indexes] *= -1

    # Multiply the negative values by 0.5
    z[z < 0] *= 0.5

    z = 1 + z / 100

    product = np.prod(z)

    val = (product ** (1 / len(z)) - 1) * 100

    print(f"random percent per day:\t {val:.2f}, \t\t250_days:\t {(product-1)*100 :.2f}")
