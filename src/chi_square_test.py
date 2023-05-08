import numpy as np
from scipy.stats import chi2_contingency

# Input the observed frequencies as a 2x2 contingency table
# For example, let's say we have the following data:
# Group 1: 50 successes out of 200 trials
# Group 2: 40 successes out of 150 trials
contingency_table = np.array([
    [9, 56],
    [4, 7]
])

# Perform the chi-square test
chi2_stat, p_value, dof, expected_freq = chi2_contingency(contingency_table)

# Print the results
print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_value)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected_freq)

# Set the significance level
alpha = 0.05

# Compare the p-value to the significance level
if p_value < alpha:
    print("There is a significant difference between the two ratios.")
else:
    print("There is no significant difference between the two ratios.")
