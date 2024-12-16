import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset, assuming the file is named 'dataset.csv'
data = pd.read_csv('dataset.csv')

# Count the occurrences of each label
label_counts = data[' Label'].value_counts()

# Set the plot style and figure size
plt.figure(figsize=(10, 6))  # Set figure size

# Create a bar plot to visualize label distribution
sns.barplot(x=label_counts.index, y=label_counts.values, linewidth=0.1)

# Add labels and title
plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Label Distribution', fontsize=14)

# Rotate x-axis labels to prevent overlap
plt.xticks(rotation=45, ha='right')

# Display count values on top of each bar
for i, v in enumerate(label_counts.values):
    plt.text(i, v + 10, str(v), ha='center', va='center', fontweight='bold', color='black')

# Add a legend to the plot
plt.legend(['Count'], loc="upper right")

# Show the plot
plt.show()