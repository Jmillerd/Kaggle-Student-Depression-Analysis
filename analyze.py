from dataclasses import dataclass, fields
from typing import Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data\Student Depression Dataset.csv')

# 2. Define Dataclass for Student Depression Record
@dataclass
class StudentRecord:
    student_id: Optional[int] = None
    gender: Optional[str] = None
    age: Optional[float] = None
    city: Optional[str] = None
    profession: Optional[str] = None
    academic_pressure: Optional[float] = None
    work_pressure: Optional[float] = None
    cgpa: Optional[float] = None
    study_satisfaction: Optional[float] = None
    job_satisfaction: Optional[float] = None
    sleep_duration: Optional[str] = None
    dietary_habits: Optional[str] = None
    degree: Optional[str] = None
    suicidal_thoughts: Optional[str] = None
    work_study_hours: Optional[float] = None
    financial_stress: Optional[float] = None
    family_history: Optional[str] = None
    depression: Optional[int] = None

# 3. Convert DataFrame Rows to List of Dataclass Instances
def dataframe_to_dataclasses(df: pd.DataFrame) -> List[StudentRecord]:
    valid_fields = {field.name for field in fields(StudentRecord)}
    return [
        StudentRecord(**{key: value for key, value in row.items() if key in valid_fields})
        for _, row in df.iterrows()
    ]

# 4. Analyze Correlation Between Sleep Habits and Depression
def analyze_sleep_depression(df: pd.DataFrame):
    sleep_mapping = {
        'Less than 5 hours': 1,
        '5-6 hours': 2,
        '7-8 hours': 3,
        'More than 8 hours': 4
    }
    df['Sleep Duration Numeric'] = df['Sleep Duration'].map(sleep_mapping)
    df_clean = df.dropna(subset=['Sleep Duration Numeric', 'Depression'])

    correlation = df_clean[['Sleep Duration Numeric', 'Depression']].corr().iloc[0, 1]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Sleep Duration Numeric', y='Depression', data=df_clean)
    plt.title(f'Correlation between Sleep Duration and Depression (r = {correlation:.2f})')
    plt.xlabel('Sleep Duration (1:<5h, 2:5-6h, 3:7-8h, 4:>8h)')
    plt.ylabel('Depression (0:No, 1:Yes)')
    plt.grid(True)
    plt.show()
    
    print(f'Correlation between Sleep Duration and Depression: {correlation:.2f}')


def analyze_job_pressure_depression(df: pd.DataFrame):

    df_clean = df.dropna(subset=['Work Pressure', 'Depression'])

    correlation = df_clean[['Work Pressure', 'Depression']].corr().iloc[0, 1]
    
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='Work Pressure', y='Depression', data=df_clean)
    plt.title(f'Correlation between Work Pressure and Depression (r = {correlation:.2f})')
    plt.xlabel('Work Pressure')
    plt.ylabel('Depression (0:No, 1:Yes)')
    plt.grid(True)
    plt.show()

def analyze_financial_stress_depression(df: pd.DataFrame):

  plt.figure(figsize=(7,5))
  sns.barplot(x='Financial Stress', y='Depression', data=df)
  plt.title('Average Depression Rate by Financial Stress')
  plt.xlabel('Financial Stress (1-5)')
  plt.ylabel('Average Depression Rate')
  plt.grid(True)
  plt.show()

def analyze_academic_pressure_depression(df: pd.DataFrame):
  
  df = df[pd.to_numeric(df['Academic Pressure'], errors='coerce').notna()]
  df['Academic Pressure'] = pd.to_numeric(df['Academic Pressure'])
  df_clean = df[df['Academic Pressure'] > 0]  # Only positive categories

  plt.figure(figsize=(7,5))
  sns.barplot(x='Academic Pressure', y='Depression', data=df_clean)
  plt.title('Average Depression Rate by Academic Pressure')
  plt.xlabel('Academic Pressure (1-5)')
  plt.ylabel('Average Depression Rate')
  plt.grid(True)
  plt.show()

def analyze_city_depression_instances(df: pd.DataFrame):
 top_cities = df['City'].value_counts().head(10).index
 df_top = df[df['City'].isin(top_cities)].copy()  # Make an explicit copy

 city_order = df_top['City'].value_counts().index  # Get city order from counts
 df_top.loc[:, 'City'] = pd.Categorical(df_top['City'], categories=city_order, ordered=True)


# Horizontal Count Plot with Depression Status
 plt.figure(figsize=(8, 6))
 sns.countplot(y='City', hue='Depression', data=df_top, order=top_cities)
 plt.title('Top 10 Cities: Students Count by Depression Status')
 plt.xlabel('Student Count')
 plt.ylabel('City')
 plt.grid(True)
 plt.legend(title='Depression', labels=['No (0)', 'Yes (1)'])
 plt.show()

def analyze_degree_depression_instances(df: pd.DataFrame):
  top_degrees = df['Degree'].value_counts().head(10).index
  df_top = df[df['Degree'].isin(top_degrees)].copy()  # Force a copy

  degree_order = df_top['Degree'].value_counts().index

  df_top.loc[:, 'Degree'] = pd.Categorical(df_top['Degree'], categories=degree_order, ordered=True)

# Horizontal Count Plot with Depression Status
  plt.figure(figsize=(8, 6))
  sns.countplot(y='Degree', hue='Depression', data=df_top, order=top_degrees)
  plt.title('Top 10 Degree: Students Count by Depression Status')
  plt.xlabel('Student Count')
  plt.ylabel('Degree')
  plt.grid(True)
  plt.legend(title='Depression', labels=['No (0)', 'Yes (1)'])
  plt.show()

def analyze_gender_based_depression(df: pd.DataFrame):
# Group Data by Gender and Depression Status
    gender_distribution = df.groupby(['Gender', 'Depression']).size().reset_index(name='Count')

# Create Separate Pie Charts for Each Gender
    for gender in gender_distribution['Gender'].unique():
        gender_data = gender_distribution[gender_distribution['Gender'] == gender]
        labels = ['No Depression (0)', 'Depression (1)']
        sizes = gender_data['Count'].values
    
        plt.figure(figsize=(6,6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title(f'{gender} Students: Depression Distribution')
        plt.show()

def analyze_health_habit_based_depresssion(df: pd.DataFrame):
   # Group Data by Dietary Habits and Depression Status
    health_habit_dist = df.groupby(['Dietary Habits', 'Depression']).size().reset_index(name='Count')

# Iterate Through Each Dietary Habit
    for habit in health_habit_dist['Dietary Habits'].unique():
        habit_data = health_habit_dist[health_habit_dist['Dietary Habits'] == habit]

        labels = ['No Depression (0)', 'Depression (1)']
        sizes = habit_data['Count'].values

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%', 
            colors=['#66b3ff','#ff9999'],  # 2 colors for 2 categories
            startangle=140
        )
        plt.axis('equal')  # Equal aspect ratio for a perfect circle
        plt.title(f'{habit} Students: Depression Distribution')
        plt.show()


# 5. Run Analysis

