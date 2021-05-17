#!/usr/bin/env python
# coding: utf-8

# In[2]:


import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd


# In[4]:


xl = pd.read_excel('Players.xlsx')
teams = pd.read_csv("teams.csv")
deliveries = pd.read_csv("deliveries.csv")
matches = pd.read_csv("matches.csv",parse_dates=['date'])
teamwise_home_and_away = pd.read_csv("teamwise_home_and_away.csv")
most_runs_average_strikerate = pd.read_csv("most_runs_average_strikerate.csv")


# In[5]:


print("No. of teams: ",teams['team1'].nunique())
teams['team1'].unique()
print(teams.info())


# In[6]:


print(deliveries.info())
deliveries.tail()


# In[7]:


print(matches.info())
matches.head()


# In[8]:


matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)

deliveries.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)


# In[9]:


print("No. of Umpires 1: ",matches['umpire1'].nunique())
print("No. of Umpires 2: ",matches['umpire2'].nunique())
print("No. of Umpires 3: ",matches['umpire3'].nunique())

ump_set1 = set(matches['umpire1'].unique())               
ump_set2 = set(matches['umpire2'].unique())
ump_set3 = set(matches['umpire3'].unique())
all_set = ump_set1.intersection(ump_set2)
all_set = all_set.intersection(ump_set3)
print("Umpires who umpired as 1st,2nd and 3rd umpires: ",all_set, len(all_set))


# In[10]:


plt.subplots(figsize=(14,6))
ax=matches['umpire1'].value_counts().plot.bar(width=0.9,color=sns.color_palette('bright',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.xlabel("Umpires", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.title("Umpires-1 who have umpired most (from highest to lowest)", fontsize=20)
plt.show()


# In[11]:


plt.subplots(figsize=(14,6))
ax=matches['umpire2'].value_counts().plot.bar(width=0.9,color=sns.color_palette('pastel',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.xlabel("Umpires", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.title("Umpires-2 who have umpired most (from highest to lowest)", fontsize=20)
plt.show()


# In[12]:


plt.subplots(figsize=(14,6))
ax=matches['umpire3'].value_counts().plot.bar(width=0.9,color=sns.color_palette('Blues'))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.xlabel("Umpires", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.title("Umpires-3 who have umpired most (from highest to lowest)", fontsize=20)
plt.show()


# In[13]:


plt.subplots(figsize=(10,6))
ax=matches['toss_winner'].value_counts().plot.bar(width=0.9,color=sns.color_palette('RdYlGn',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
plt.title("Teams that won the toss (from highest to lowest)", fontsize=20)
plt.xlabel("Teams", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()


# In[14]:


plt.subplots(figsize=(10,6))
sns.countplot(x='Season',hue='toss_decision',data=matches ,palette=sns.color_palette('bright'))
plt.title("Decision to field or bat across seasons")
plt.show()


# In[15]:


plt.subplots(figsize=(10,6))
sns.countplot(x='Season',data=matches,palette=sns.color_palette('colorblind'))  #countplot automatically counts the frequency of an item
plt.title("Number of matches played across Seasons")
plt.show()


# In[16]:


pm = matches.groupby(['player_of_match'])['id'].count().reset_index('player_of_match').rename(columns={'player_of_match':'player','id':'count'})#.sort_values(ascending=False)
pm = pm.sort_values(by="count",ascending=False)
top_pm=pm[:10]

fig = go.Figure(data=[go.Scatter(
    x=top_pm['player'], y=top_pm['count'],
    mode='markers',
    marker=dict(
        color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',
               'rgb(44, 160, 101)', 'rgb(255, 65, 54)','rgb(92, 65, 54)','rgb(150, 65, 54)','rgb(30, 165, 54)',
              'rgb(100, 180, 120)', 'rgb(200, 90, 89)', 'rgb(225, 78, 124)'],
        opacity=[1, 0.9, 0.8,0.7, 0.6,0.5,0.45,0.4,0.35,0.3],
        size=[100, 90, 80, 70,60,50,40,30,20,10],
    )
)])
fig.update_layout(
    title="Players who recieved 'Player of Match' Award most",
    xaxis=dict(
        title='Players',        
    ),
    yaxis=dict(
        title='Number',       
    ))
fig.show()


# In[17]:


print("Total number of Cities played: ",matches['city'].nunique())
print("Total number of Venues played: ",matches['venue'].nunique())


# In[18]:


plt.subplots(figsize=(10,15))
ax = matches['venue'].value_counts().sort_values(ascending=True).plot.barh(width=.9,color=sns.color_palette('inferno',40))
ax.set_xlabel('Grounds')
ax.set_ylabel('count')
plt.title("Venues played (from most to least)")
plt.show()


# In[19]:


cities = matches.groupby(['Season','city'])['id'].agg('count').reset_index()
cities.rename(columns={'id':'count'}, inplace=True)

fig = px.bar(cities, x="city", y="count", color='Season')
fig.show()

print(matches.columns)


# In[20]:


not_same = matches[matches['toss_winner'] != matches['winner']]
same = matches[matches['toss_winner'] == matches['winner']]
print("Percentage of matches where toss winner is not same as winner: ",round(not_same.shape[0]/matches.shape[0],2) *100)
print("Percentage of matches where toss winner is same as winner: ", round(same.shape[0]/matches.shape[0],2) * 100)
toss_winner = pd.DataFrame({'result':['Yes','No'],'per':[same.shape[0], not_same.shape[0]] })
print("*" * 70)
field = matches[matches['toss_decision'] == 'field']
bat = matches[matches['toss_decision'] == 'bat']
print("Percentage of matches where toss decision is 'field': ",round(field.shape[0]/matches.shape[0],2) *100)
print("Percentage of matches where toss decision is 'bat': ",round(bat.shape[0]/matches.shape[0],2) *100)
print("*" * 70)
normal = matches[matches['result'] == 'normal']
tie = matches[matches['result'] == 'tie']
no_result = matches[matches['result'] == 'no result']
print("Percentage of matches where result is 'normal': ",round(normal.shape[0]/matches.shape[0],2) *100)
print("Percentage of matches where result is 'tie': ",round(tie.shape[0]/matches.shape[0],2) *100)
print("Percentage of matches where result is 'no result': ",round(no_result.shape[0]/matches.shape[0],2) *100)
result = pd.DataFrame({'Result':['Normal','Tie','No Result'],'per':[normal.shape[0], tie.shape[0], no_result.shape[0]] })
print("*" * 70)
dl_applied_no = matches[matches['dl_applied'] == 0]
dl_applied_yes = matches[matches['dl_applied'] == 1]
dl = pd.DataFrame({'dl_applied':['yes','no'],'per':[dl_applied_yes.shape[0], dl_applied_no.shape[0]] })
print("Percentage of matches where Duckworth–Lewis–Stern method (DLS) is applied : ",round(dl_applied_yes.shape[0]/matches.shape[0],2) *100)
print("Percentage of matches where Duckworth–Lewis–Stern method (DLS) is not applied : ",round(dl_applied_no.shape[0]/matches.shape[0],2) *100)

fig = px.pie(toss_winner, values='per', names='result', color='result', title='Is Match winner same as toss winner?'
             ,color_discrete_map={'Yes':'#F0FFFF',
                                 'No':'#B0E0E6' })
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
fig = px.pie(dl, values='per', names='dl_applied', title='Percentage of matches where Duckworth–Lewis–Stern method (DLS) is applied', color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

labels = result['Result']
values = result['per']
fig = go.Figure(data=[go.Pie(labels=labels,title='Result of matches', values=values, pull=[0, 0.2, 0.1])])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[21]:


matches['date'].min(), matches['date'].max()


# In[22]:


teamwise_home_and_away.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)
print(teamwise_home_and_away.info())
teamwise_home_and_away.head()


# In[23]:


fig = go.Figure(data=[
    go.Bar(name='Home Win', x=teamwise_home_and_away['team'], y=teamwise_home_and_away['home_win_percentage']),
    go.Bar(name='Away Win', x=teamwise_home_and_away['team'], y=teamwise_home_and_away['away_win_percentage'])
])

fig.update_layout(barmode='group',title="Team wise - Home/Away wins")
fig.show()


# In[24]:


fig = go.Figure(data=[
    go.Bar(name='Home Win', x=teamwise_home_and_away['team'], y=teamwise_home_and_away['home_win_percentage']),
    go.Bar(name='Away Win', x=teamwise_home_and_away['team'], y=teamwise_home_and_away['away_win_percentage'])
])

fig.update_layout(barmode='group',title="Team wise - Home/Away wins")
fig.show()


# In[25]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

top15 = most_runs_average_strikerate[:15]

fig = go.Figure()
fig.add_trace(go.Bar(
   y=top15['batsman'],
    x=top15['out'],
    name='No. of Matches',
    orientation='h',
    marker=dict(
        color='rgba(80, 100, 67, 0.6)',
        line=dict(color='rgba(8, 1, 212, 1.0)', width=3)
    )
))
fig.add_trace(go.Bar(
   y=top15['batsman'],
    x=top15['strikerate'],
    name='Strike Rate',
    orientation='h',
    marker=dict(
        color='rgba(8, 1, 212, 0.6)',
        line=dict(color='rgba(8, 1, 212, 1.0)', width=3)
    )
))
fig.add_trace(go.Bar(
   y=top15['batsman'],
    x=top15['average'],
    name='Average Runs',
    orientation='h',
    marker=dict(
        color='rgba(158, 5, 19, 0.6)',
        line=dict(color='rgba(158, 5, 19, 1.0)', width=3)
    )
))

fig.update_layout(barmode='stack',title="Players - No. of matches, Strike Rate, Average Runs")
fig.show()


# In[26]:


plt.subplots(figsize=(8,6))
b = deliveries.groupby(['batsman'])['batsman_runs'].sum()#.sort_values('batsman_runs')
b = b.sort_values(ascending=False)
b[100:200]
ax=b.sort_values(ascending=False)[:10].plot.bar(width=0.8,color=sns.color_palette('husl',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.1, p.get_height()+50),fontsize=15)
plt.show()


# In[27]:


top_batsman = deliveries.groupby(['batsman','batsman_runs'])['total_runs'].count().reset_index()
top_batsman = top_batsman.pivot('batsman','batsman_runs','total_runs')
fig,ax=plt.subplots(4,2,figsize=(24,16))
top_batsman[1].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[0,0],color='#45ff45',width=0.8)
ax[0,0].set_title("Most 1's")
ax[0,0].set_ylabel('')
top_batsman[2].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[0,1],color='#df6dfd',width=0.8)
ax[0,1].set_title("Most 2's")
ax[0,1].set_ylabel('')
top_batsman[4].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[1,0],color='#fbca5f',width=0.8)
ax[1,0].set_title("Most 4's")
ax[1,0].set_ylabel('')
top_batsman[6].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[1,1],color='#ffff00',width=0.8)
ax[1,1].set_title("Most 6's")
ax[1,1].set_ylabel('')
top_batsman[0].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[2,0],color='#abcd00',width=0.8)
ax[2,0].set_title("Most 0's")
ax[2,0].set_ylabel('')
top_batsman[7].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[2,1],color='#f0debb',width=0.8)
ax[2,1].set_title("Most 7's")
ax[2,1].set_ylabel('')
top_batsman[3].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[3,0],color='#778899',width=0.8)
ax[3,0].set_title("Most 3's")
ax[3,0].set_ylabel('')
top_batsman[5].sort_values(ascending=False)[:5].plot(kind='barh',ax=ax[3,1],color='#F0E68C',width=0.8)
ax[3,1].set_title("Most 5's")
ax[3,1].set_ylabel('')
plt.show()


# In[28]:


top_scorers = deliveries.groupby(["match_id", "batsman","batting_team"])["batsman_runs"].sum().reset_index()
top_scorers.sort_values('batsman_runs', ascending=0).head(10)
top_scorers.nlargest(10,'batsman_runs')


# In[29]:


batsmen = matches[['id','Season']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
season=batsmen.groupby(['Season','batsman'])['total_runs'].sum().reset_index()
season.set_index('Season').plot(marker='*')
plt.gcf().set_size_inches(10,8)
plt.title('Total Runs Across the Seasons')
plt.show()


# In[30]:


batsmen = matches[['id','Season']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
#merging the matches and delivery dataframe by referencing the id and match_id columns respectively
season=batsmen.groupby(['Season'])['total_runs'].sum().reset_index()
season.set_index('Season').plot(marker='*')
plt.gcf().set_size_inches(10,8)
plt.title('Total Runs Across the Seasons')
plt.show()


# In[31]:


batsmen = matches[['id','Season']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
season=batsmen.groupby(['Season','batsman'])['batsman_runs'].sum().reset_index()
s = season.groupby(['Season','batsman'])['batsman_runs'].sum().unstack().T
season.set_index('Season').plot(marker='*')
plt.gcf().set_size_inches(10,8)
plt.title('Total Runs Across the Seasons')
plt.show()


# In[32]:


men = batsmen.groupby(['Season','batsman'])['batsman_runs'].sum().reset_index()
men = men.groupby(['Season','batsman'])['batsman_runs'].sum().unstack().T
men['Total'] = men.sum(axis=1)
men = men.sort_values(by='Total',ascending=False)[:1]
men.drop('Total',axis=1,inplace=True)
men.T.plot(color=['red'])
fig=plt.gcf()
fig.set_size_inches(16,8)
plt.show()


# In[33]:


men = batsmen.groupby(['Season','batsman'])['batsman_runs'].sum().reset_index()
men = men.groupby(['Season','batsman'])['batsman_runs'].sum().unstack().T
men['Total'] = men.sum(axis=1)
men = men.sort_values(by='Total',ascending=False)[:5]
men.drop('Total',axis=1,inplace=True)
men.T.plot(color=['red','skyblue','#772272','black','limegreen'],marker='*')
fig=plt.gcf()
fig.set_size_inches(16,8)
plt.show()


# In[34]:


Season_boundaries=batsmen.groupby("Season")["batsman_runs"].agg(lambda x: (x==6).sum()).reset_index()
a=batsmen.groupby("Season")["batsman_runs"].agg(lambda x: (x==4).sum()).reset_index()
Season_boundaries=Season_boundaries.merge(a,left_on='Season',right_on='Season',how='left')
Season_boundaries=Season_boundaries.rename(columns={'batsman_runs_x':'6"s','batsman_runs_y':'4"s'})
Season_boundaries.set_index('Season')[['6"s','4"s']].plot(marker='o')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# In[35]:


matches_played_byteams=pd.concat([matches['team1'],matches['team2']])
matches_played_byteams=matches_played_byteams.value_counts().reset_index()
matches_played_byteams.columns=['Team','Total Matches']
matches_played_byteams['wins']=matches['winner'].value_counts().reset_index()['winner']
matches_played_byteams.set_index('Team',inplace=True)
runs_per_over = deliveries.pivot_table(index=['over'],columns='batting_team',values='total_runs',aggfunc=sum)
runs_per_over[(matches_played_byteams[matches_played_byteams['Total Matches']>50].index)].plot(color=["blue", "red", "#Ffb6b2", "green",'brown','yellow','#6666ff','black','#FFA500']) #plotting graphs for teams that have played more than 100 matches
x=[x for x in range(1,21,1)]
plt.xticks(x)
plt.ylabel('total runs scored')
fig=plt.gcf()
fig.set_size_inches(16,10)
plt.show()


# In[36]:


high_scores=deliveries.groupby(['match_id', 'inning','batting_team','bowling_team'])['total_runs'].sum().reset_index() 
high_scores=high_scores[high_scores['total_runs']>=200]
high_scores.nlargest(10,'total_runs')


# In[ ]:




