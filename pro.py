import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

# with model_training:
# st.header('Time to train my model!')
# st.text('ddd')

header=st.container()
dataset=st.container()
features=st.container()
model_training=st.container()
#
with header:
    st.title('Welcome to my awesome data project!')
    st.text('In this project I took into the Salaries distributed in various ways')
with dataset:
    st.header('Salary by Job Title and Country dataset')
    st.text('I found this dataset on kaggle.com')
    st.code('''df=pd.read_csv('Salary 2.csv')
df''')
    df = pd.read_csv('Salary 2.csv')
    st.write(df)
    st.header('Let us describe some main features')
    st.code('''df.describe()''')
    st.write(df.describe())
    st.header('Data Preprocessing')
    st.text('null values')
    st.write(df.isnull().sum().sum())
    st.code('''#Let’s check Nan 
df.isnull().sum().sum()
#And I wanna to delete them
df = df.dropna()
df.isnull().sum().sum()''')
    st.header('Now I want to introduce some Jobs and their Salaries')
    df = pd.read_csv('Salary 2.csv')
    avg_job = df[['Job Title', 'Salary']].groupby(by='Job Title', as_index=False).median().sort_values(by='Salary',
                                                                                                       ascending=False)
    # This sorts the median salary in descending order
    fig = px.bar(avg_job.iloc[:20, :], x='Salary', y='Job Title', title='<b>Salary by Profession<b>', color='Salary')
    fig.update_layout(width=800, height=1000, template='simple_white')
    st.plotly_chart(fig,theme=None)
    st.code('''df=pd.read_csv('Salary 2.csv')
avg_job = df[['Job Title', 'Salary']].groupby(by='Job Title', as_index=False).median().sort_values(by='Salary', ascending=False)
#This sorts the median salary in descending order, so the highest paying jobs come first.
fig = px.bar(avg_job.iloc[:20,:], x='Salary', y='Job Title', title='<b>Salary by Profession<b>',color='Salary')
fig.update_layout(width=800, height=1000,template='simple_white')
fig.show()
''')
    st.header('I also want to remove duplicates that may interfere with the accuracy of the data')
    st.text('Duplicates and shape')
    st.write(df.duplicated().sum(),df.shape)
    df.drop_duplicates(inplace=True)
    st.code('''#For accuracy, repetitions should be removed
df.drop_duplicates(inplace=True)
df.head()''')
    st.code('''df.shape''')
    st.write(df.duplicated().sum(),df.shape)
    # In the same way, I need to focus on at least 10 people for maximum accuracy
    value_counts = df['Job Title'].value_counts()
    to_remove = value_counts[value_counts < 10].index
    df = df[~df['Job Title'].isin(to_remove)]
    df['Job Title'].value_counts()
    df
    st.code('''#In the same way, I need to focus on at least 10 people for maximum accuracy
value_counts = df['Job Title'].value_counts()
to_remove = value_counts[value_counts<10].index
df = df[~df['Job Title'].isin(to_remove)]
df['Job Title'].value_counts()
df''')
    st.write(df.head())
    st.text('Time to find the highest paying jobs')
    df2 = df.groupby('Job Title')
    highest_paid_jobs = df2.max().head(5)
    sorted_highest_paid_jobs = highest_paid_jobs.sort_values(by='Salary', ascending=False).reset_index()
    sorted_highest_paid_jobs
    st.code('''import pandas as pd
df2 = df.groupby('Job Title')
highest_paid_jobs = df2.max().head(5)
sorted_highest_paid_jobs = highest_paid_jobs.sort_values(by='Salary', ascending=False).reset_index()
sorted_highest_paid_jobs''')
    # Let's see what the highest paying jobs are
    import pandas as pd
    import plotly.express as px

    fig = px.scatter(sorted_highest_paid_jobs, x='Job Title', y='Salary',
                     color='Job Title',   # Each job title gets its own marker color
                     size='Salary',
                     size_max=40, title='Distribution of Salaries',template='simple_white')

    fig.update_layout(width=1000, height=600)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig,theme=None)
    st.code('''#Let's see what the highest paying jobs are
import pandas as pd
import plotly.express as px
fig = px.scatter(sorted_highest_paid_jobs, x='Job Title', y='Salary',
                 color='Job Title', symbol='Job Title',  # Each job title gets its own marker color
                 size='Salary', 
                 size_max=40,title='Boxplot Distribution of Salaries')

fig.update_layout(width=1000, height=600)
fig.update_layout(showlegend=False)
fig.show()
''')
    st.header('Its hypothesis time!')
    st.text('It appears that improving someones salary is more dependent on their education level. '
            'Lets see if this applies to the dataset of the top 30 highest paying job titles.')
    highest_paid_jobs = df2.max().head(30)
    fig = px.scatter(highest_paid_jobs, x="Salary", y="Years of Experience", size="Education Level",
                     color="Education Level", marginal_x="violin", marginal_y="histogram", trendline="ols",
                     trendline_color_override='#27187e')
    fig.update_layout(width=800, height=800, template='simple_white')
    st.plotly_chart(fig,theme=None)
    st.code('''highest_paid_jobs = df2.max().head(30) 
fig = px.scatter(highest_paid_jobs, x="Salary", y="Years of Experience",size="Education Level", color="Education Level", marginal_x="violin",marginal_y="histogram",trendline="ols",
                trendline_color_override='#27187e')
fig.update_layout(width=1000, height=800,template='simple_white')
fig.show()
#It looks like boosting someone's salary is more dependent on their education level.''')
    st.text('Yeah!Its excatly true')
    st.text('Now let us check what about female and male allocation. '
            'I have a feeling that there will be a distribution among the races')
    color_mapping = {'Male': '#ff6d00', 'Female': '#7209b7'}
    fig = px.box(
        df,
        x="Race",
        y="Salary", color_discrete_map=color_mapping,
        facet_col="Gender", template='simple_white', color="Gender",
        title='<b>Dependence of Male and Female Salaries and their Race</b><br><sup><i>Plot 3<i></sup>')
    for i in range(len(fig.data)):
        fig.update_xaxes(title_text="", row=1, col=i + 1)
        fig.add_annotation(
            dict(
                x=0.5,
                y=-0.50,
                xref='paper',
                yref='paper',
                text='<b><i>Race<i><b>',
                showarrow=False,
                font=dict(size=14.5),
            ))
    st.plotly_chart(fig, theme=None)
    fig.update_layout(width=1000, height=600)
    st.code('''color_mapping = {'Male': '#ff6d00', 'Female': '#7209b7'}
fig = px.box(
    df,
    x="Race",
    y="Salary",color_discrete_map=color_mapping,
    facet_col="Gender",template='simple_white', color="Gender",title='<b>Dependence of Male and Female Salaries and their Race</b><br><sup><i>Plot 3<i></sup>')
for i in range(len(fig.data)):
    fig.update_xaxes(title_text="", row=1, col=i+1)
    fig.add_annotation(
    dict(
        x=0.5,
        y=-0.50,
        xref='paper',
        yref='paper',
        text='<b><i>Race<i><b>',
        showarrow=False,
        font=dict(size=14.5),
    ))
    In addition, mens salaries are a bit inelastic to race compared to female salaries.
fig.show()''')
    st.text('Thus, mens salaries are inelastic to race level compared to female salaries.The male and female distributions are about the same')
    st.text('Maybe we can check it more in detain through a histogram? And compare the other indicators')
    fig = px.histogram(
        df,
        x="Age",
        y="Salary",
        color="Gender",
        facet_col="Education Level", color_discrete_map={'Male': '#ff9e00', 'Female': '#9d4edd'},
        histfunc='avg', marginal="box",
        title='<b>Dependence of Male and Female Salaries on Education Levels and Age</b><br><sup><i>Plot 3<i></sup>',
        template='simple_white',
        nbins=40,  # Number of bins for the histograms
    )

    # Update layout for better presentation
    fig.update_layout(
        barmode="overlay",
        showlegend=True)
    for i in range(len(fig.data)):
        fig.update_xaxes(title_text="", row=1, col=i + 1)
    fig.add_annotation(
        dict(
            x=0.5,
            y=-0.30,
            xref='paper',
            yref='paper',
            text='<b><i>Age<i><b>',
            showarrow=False,
            font=dict(size=11.5),
        ))
    fig.update_layout(width=900, height=500)
    # Show the figure
    st.plotly_chart(fig, theme=None)
    st.code('''fig = px.histogram(
    df,
    x="Age",
    y="Salary",
    color="Gender",
    facet_col="Education Level",
    color_discrete_map={'Male': '#ff9e00', 'Female':'#9d4edd'},
    histfunc='avg', marginal="box",
title='<b>Dependence of Male and Female Salaries on Education Levels and Age</b><br><sup><i>Plot 3<i></sup>',template='simple_white',
    nbins=40,
)

# Update layout for better presentation
fig.update_layout(
    barmode="overlay",
    showlegend=True)
for i in range(len(fig.data)):
    fig.update_xaxes(title_text="", row=1, col=i+1)
fig.add_annotation(
    dict(
        x=0.5,
        y=-0.30,
        xref='paper',
        yref='paper',
        text='<b><i>Age<i><b>',
        showarrow=False,
        font=dict(size=11.5),
    ))
# Show the figure
fig.show()''')
    st.text('#Lets look at the distribution of wages in different countries')
    import plotly.express as px
    df2 = pd.read_csv('Salary 2.csv')
    fig = px.scatter(df2, x="Age", y="Salary", color="Country", marginal_x="histogram",
                     color_discrete_map={'UK': '#3a86ff', 'USA': '#ffbe0b', 'Canada': '#8338ec', 'China': '#fb5607',
                                         'Australia': '#ff006e'},
                     title='<b>Dependence Salary by Country</b><br><sup><i>Plot 3<i></sup>', template='simple_white',
                     trendline="ols", marginal_y="box").update_layout(xaxis_type="log")
    fig.update_layout(width=1000, height=600)
    st.plotly_chart(fig, theme=None)
    st.code('''import plotly.express as px
fig = px.scatter(df2, x="Age", y="Salary", color="Country",marginal_x="histogram", 
                 color_discrete_map={'UK': '#3a86ff', 'USA':'#ffbe0b', 'Canada': '#8338ec','China': '#fb5607','Australia': '#ff006e'},
                 title='<b>Dependence Salary by Country</b><br><sup><i>Plot 3<i></sup>',template='simple_white',
                 trendline="ols",marginal_y="box").update_layout(xaxis_type="log")
fig.update_layout(title_text='Scatter Plot of Salary by Country', width=1000, height=600)
fig.show()''')
    st.text('It seems possible to conclude that regardless of country everywhere age directly affects salaries')
    st.text('Is there any discrimination?')
    import pandas as pd
    import plotly.express as px

    df2 = pd.read_csv('Salary 2.csv')
    # Aggregate the data by taking the meadian salary for each race
    df_aggregated = df2.groupby('Race', as_index=False)['Salary'].median().sort_values("Salary", ascending=False)
    fig = px.bar(df_aggregated, x='Race', y='Salary', color='Race',
                 color_discrete_map={'Black': '#3a86ff', 'Korean': '#ffbe0b', 'Mixed': '#8338ec', 'White': '#fb5607',
                                     'Asian': '#ff006e', 'Australian': '#7678ed',
                                     'African American': '#2d00f7', 'Chinese': '#b8b8ff', 'Welsh': '#da667b',
                                     'Hispanic': '#f2b5d4'},
                 title='<b>Relationship between Race and median Salary</b><br><sup><i>Plot 1<i></sup>',
                 text=df_aggregated['Salary'].apply(lambda x: f'{x / 1000:.1f}'), template='simple_white')

    fig.update_layout(width=800, height=500)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, theme=None)
    # There doesn't seem to be any discrimination here
    st.code('''import pandas as pd
import plotly.express as px
df2=pd.read_csv('Salary 2.csv')
# Aggregate the data by taking the meadian salary for each race
df_aggregated = df2.groupby('Race', as_index=False)['Salary'].median().sort_values("Salary", ascending=False)
fig = px.bar(df_aggregated, x='Race', y='Salary', color='Race',
             color_discrete_map={'Black': '#3a86ff', 'Korean':'#ffbe0b', 'Mixed': '#8338ec','White': '#fb5607','Asian': '#ff006e','Australian': '#7678ed',
                                'African American': '#2d00f7','Chinese': '#b8b8ff','Welsh': '#da667b','Hispanic': '#f2b5d4'},
             title='<b>Relationship between Race and median Salary</b><br><sup><i>Plot 1<i></sup>',text = df_aggregated['Salary'].apply(lambda x: f'{x/1000:.1f}'), template='simple_white')

fig.update_layout(width=800, height=500)
fig.update_layout(showlegend=False)
fig.show()
''')
    st.text('Consider some of the dependencies, but for the seniors')
    import plotly.express as px
    from plotly.subplots import make_subplots

    df2 = pd.read_csv('Salary 2.csv')
    fig = make_subplots(
        rows=2,
        cols=2, specs=[[{}, {}],
                       [{"colspan": 2}, None]],
        subplot_titles=[
            'Years of Experience',
            'Age',
            'Overall salaries'
        ]
    )
    # Scatter plot for Years of Experience vs Salary
    scatter_1 = px.scatter(
        df,
        x="Years of Experience",
        y="Salary",
        color="Senior",
        title='Years of Experience vs Salary')
    fig.add_trace(scatter_1['data'][0], row=1, col=1)
    # Scatter plot for Age vs Salary
    scatter_2 = px.scatter(
        df2,
        x="Age",
        y="Salary", color="Senior",  # Specify color
        title='Age vs Salary'
    )
    fig.add_trace(scatter_2['data'][0], row=1, col=2)

    # Histograms for All Salaries and Senior Salaries
    histogram_all_salaries = px.histogram(df2, x="Salary", nbins=30, color_discrete_sequence=['#560bad'],
                                          histfunc='count', title='All Salaries')
    histogram_senior_salaries = px.histogram(df[df['Senior'] == 1], x="Salary", nbins=30,
                                             color_discrete_sequence=['#fcf300'], histfunc='count',
                                             title='Senior Salaries')

    fig.add_trace(histogram_all_salaries['data'][0], row=2, col=1)
    fig.add_trace(histogram_senior_salaries['data'][0], row=2, col=1)
    fig.update_layout(
        title_text='<b>Seniors salaries per age, per their experience and their salaries in total</b>',
        height=700,
        template='simple_white',
        barmode='overlay'
    )
    st.plotly_chart(fig, theme=None)
#
#st.plotly_chart(fig, theme=None)
    st.code('''import plotly.express as px
from plotly.subplots import make_subplots
df2 = pd.read_csv('Salary 2.csv')
fig = make_subplots(
    rows=2,
    cols=2,specs=[[{}, {}],
           [{"colspan": 2}, None]],
    subplot_titles=[
        'Years of Experience',
        'Age',
        'Overall salaries'
    ]
)
# Scatter plot for Years of Experience vs Salary
scatter_1 = px.scatter(
    df,
    x="Years of Experience",
    y="Salary",
    color="Senior", 
    title='Years of Experience vs Salary')
fig.add_trace(scatter_1['data'][0], row=1, col=1)
# Scatter plot for Age vs Salary
scatter_2 = px.scatter(
    df2,
    x="Age",
    y="Salary", color="Senior",
    color_discrete_map=senior_color_mapping, # Specify color
    title='Age vs Salary'
)
fig.add_trace(scatter_2['data'][0], row=1, col=2)

# Histograms for All Salaries and Senior Salaries
histogram_all_salaries = px.histogram(df2, x="Salary", nbins=30, color_discrete_sequence=['#560bad'], histfunc='count', title='All Salaries')
histogram_senior_salaries = px.histogram(df[df['Senior'] == 1], x="Salary", nbins=30, color_discrete_sequence=['#fcf300'], histfunc='count', title='Senior Salaries')

fig.add_trace(histogram_all_salaries['data'][0], row=2, col=1)
fig.add_trace(histogram_senior_salaries['data'][0], row=2, col=1)
fig.update_layout(
    title_text='<b>Seniors salaries per age, per their experience and their salaries in total</b>',
    height=700,
    template='simple_white',
    barmode='overlay'
)
fig.show()
''')
    import plotly.express as px

    fig = px.box(df2, x='Salary', y='Race', color='Senior', notched=True,
                 title='<b>Salaries of Seniors among different races<b>')
    fig.update_layout(width=600, height=600)
    st.plotly_chart(fig, theme=None)
    st.code('''import plotly.express as px
fig = px.box(df2, x='Salary', y='Race', color='Senior', notched=True,
             title='<b>Salaries of seniors among different races<b>')
fig.update_layout(width=1000, height=600)
fig.show()
''')
    st.header('I needed to add some columns to be able to count for data cleanup.'
              'I decided to do a column on starting a career')
    st.code('''
    df['Career_launch']=df['Age']-df['Years of Experience']
    df.head()''')
    df['Career_launch'] = df['Age'] - df['Years of Experience']
    st.write(df.head())
    st.text('Let us consider some dependence on this indicator')
    fig = px.histogram(df, x='Career_launch', y='Salary', template='simple_white', color_discrete_sequence=['#aeb8fe'],
                       title='<b>the relationship between career start and salary<b>')
    st.plotly_chart(fig, theme=None)
    st.code('''fig = px.histogram(df, x='Career_launch', y='Salary',template='simple_white',color_discrete_sequence=['#aeb8fe'],
                  title='<b>the relationship between career start and salary<b>')
fig.show()''')
    st.text('Also I add a new column with Gender')
    df2 = pd.get_dummies(df, columns=["Gender"])
    df2.head()
    st.code('''df2 = pd.get_dummies(df, columns = ["Gender"])
df2.head()''')
    st.text('And produce a new pie chart')
    import plotly.express as px

    male_count = df2["Gender_Male"].sum()
    female_count = df2["Gender_Female"].sum()

    labels = ["Male", "Female"]
    sizes = [male_count, female_count]
    colors = ["lightblue", '#f5cac3']

    fig = px.pie(
        values=sizes, names=labels, color=labels, template='simple_white',
        color_discrete_map=dict(zip(labels, colors)),
        title="<b>Gender Distribution<b>",
    )

    fig.update_traces(textinfo="percent+label", textfont_size=14)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, theme=None)
    st.code('''import plotly.express as px
male_count = df2["Gender_Male"].sum()
female_count = df2["Gender_Female"].sum()

labels = ["Male", "Female"]
sizes = [male_count, female_count]
colors = ["lightblue", '#f5cac3']

fig = px.pie(
    values=sizes,names=labels, color=labels, template='simple_white',
    color_discrete_map=dict(zip(labels, colors)),
    title="<b>Gender Distribution<b>",
)

fig.update_traces(textinfo="percent+label", textfont_size=14)
fig.update_layout(showlegend=False)
fig.show()''')
    st.text('Finally let us count and look at how many women and men there are in general and their distribution by education')
    import pandas as pd
    import plotly.express as px
    df = pd.read_csv('Salary 2.csv')
    avg_counts = df.groupby(['Gender', 'Education Level']).size().reset_index(name='Average Count')
    fig = px.bar(avg_counts, x='Gender', y='Average Count', color='Education Level',
                 labels={'Education Level': 'Education'},
                 title='<b>Average Education Level Counts by Gender<b>')
    fig.update_layout(width=800, height=500)
    st.plotly_chart(fig, theme=None)
    # Let's look at how many women and men there are in general and their distribution by education
    st.code('''import pandas as pd
import plotly.express as px
df = pd.read_csv('Salary 2.csv')
avg_counts = df.groupby(['Gender', 'Education Level']).size().reset_index(name='Average Count')
fig = px.bar(avg_counts, x='Gender', y='Average Count', color='Education Level',
             labels={'Education Level': 'Education'},
             title='<b>Average Education Level Counts by Gender<b>')
fig.update_layout(width=800, height=500)
fig.show()
''')
    st.header('Conclusion')
    '''On the basis of the dataset provided, it can be concluded that the salary depends significantly on the level of education and gender. '
            'And also for the start of a career. Seniors are influenced by factors such as age and experience. The most high-tailoring professions were: Date Scientist, Date Analyst, Content Manager'''
    st.text('Thank you!')
