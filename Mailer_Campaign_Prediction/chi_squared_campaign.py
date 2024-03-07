import pandas as pd
from scipy.stats import chi2_contingency, chi2

# import the data
campaign_data = pd.read_excel('./data/grocery_database.xlsx', sheet_name='campaign_data')

# remove control data
campaign_data = campaign_data.loc[campaign_data['mailer_type'] != 'Control']

# view observed values
observed_values = pd.crosstab(campaign_data['mailer_type'], campaign_data['signup_flag']).values

# mailer signup rate
mailer1_rate = observed_values[0, 1] / (observed_values[0, 0] + observed_values[0, 1])
mailer2_rate = observed_values[1, 1] / (observed_values[1, 0] + observed_values[1, 1])

# hypothesis stated
null_hypothesis = 'There is no relationship between mailer type and signup rate. They are independent.'
alternative_hypothesis = 'There is a relationship between mailer type and signup rate. They are not independent.'
acceptance_criteria = 0.05

# calculate the chi-squared statistic
chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction=False)
print('Chi-squared Statistic:', chi2_statistic)
print('P-value:', p_value)

# find the critical value
critical_value = chi2.ppf(1 - acceptance_criteria, dof)
print('Critical Value:', critical_value)

# print the chi2 results
if chi2_statistic >= critical_value:
    print(f'As our chi-squared statistic of {chi2_statistic} is greater than our critical value of {critical_value} - we reject the null hypothesis, and conclude that: {alternative_hypothesis}')
else:
    print(f'As our chi-squared statistic of {chi2_statistic} is less than our critical value of {critical_value} - we fail to reject the null hypothesis, and conclude that: {null_hypothesis}')    
    
    
# print the p-value results
if p_value <= acceptance_criteria:
    print(f'As our p-value of {p_value} is less than our acceptance criteria of {acceptance_criteria} - we reject the null hypothesis, and conclude that: {alternative_hypothesis}')
else:
    print(f'As our p-value of {p_value} is greater than our acceptance criteria of {acceptance_criteria}  - we fail to reject the null hypothesis, and conclude that: {null_hypothesis}')    
    