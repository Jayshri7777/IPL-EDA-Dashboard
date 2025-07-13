import streamlit as st
import pandas as pd
import plotly.express as px
import pickle


st.set_page_config(page_title="IPL EDA Dashboard", layout="wide")
st.title("🏏 IPL EDA Dashboard")



@st.cache_data(show_spinner=True)
def load_data():
    matches = pd.read_csv('Data/matches.csv')
    deliveries = pd.read_csv('Data/deliveries.csv')
    return matches, deliveries

matches, deliveries = load_data()


#sidebar
seasons = sorted(matches['season'].unique())
selected_seasons = st.sidebar.multiselect("Filter by Seasons", seasons, default=seasons)

teams = sorted(matches['team1'].unique())
selected_team = st.sidebar.selectbox("Select Team for Player Stats", teams)

matches_f = matches[matches['season'].isin(selected_seasons)].copy()
deliveries_f = deliveries[deliveries['match_id'].isin(matches_f['id'])].copy()

#mainTabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Team Overview",
    "Toss vs Match Win analysis",
    "Head to Head",
    "Player of the Match",
    "Data Preview",
    "Match Prediction"
])

#tab1-team overview


with tab1:
    st.header(f"{selected_team} - Key Player Metrics")

    team_bat = deliveries_f[deliveries_f['batting_team'] == selected_team]
    top_runs = (team_bat.groupby('batsman')['batsman_runs'].sum()
                       .sort_values(ascending=False).head(10)
                       .reset_index(name='Runs'))
    fig_runs = px.bar(top_runs, x='batsman', y='Runs', title='Top 10 Run‑Scorers', text_auto=True)
    st.plotly_chart(fig_runs, use_container_width=True)
    
    

    wicket_kinds = ['caught', 'bowled', 'lbw', 'stumped', 'hit wicket', 'caught and bowled',
                    'retired hurt', 'obstructing the field']
    team_bowl = deliveries_f[deliveries_f['bowling_team'] == selected_team]
    wkts = (team_bowl[team_bowl['dismissal_kind'].isin(wicket_kinds)]
                 .groupby('bowler')['dismissal_kind'].count()
                 .sort_values(ascending=False).head(10)
                 .reset_index(name='Wickets'))
    fig_wkts = px.bar(wkts, x='bowler', y='Wickets', title='Top 10 Wicket‑Takers', text_auto=True)
    st.plotly_chart(fig_wkts, use_container_width=True)
    
    
    

    bat_counts = team_bat.groupby('batsman').agg({'batsman_runs':'sum', 'ball':'count'}).reset_index()
    bat_counts['Strike Rate'] = (bat_counts['batsman_runs'] / bat_counts['ball']) * 100
    top_sr = bat_counts[bat_counts['ball'] > 60].sort_values('Strike Rate', ascending=False).head(10)
    fig_sr = px.bar(top_sr, x='batsman', y='Strike Rate', title='Best Strike Rates (>60 balls faced)', text_auto='.2f')
    st.plotly_chart(fig_sr, use_container_width=True)
    



    bowl_counts = team_bowl.groupby('bowler').agg({'total_runs':'sum', 'ball':'count'}).reset_index()
    bowl_counts['Overs'] = bowl_counts['ball'] / 6
    bowl_counts['Economy'] = bowl_counts['total_runs'] / bowl_counts['Overs']
    best_econ = bowl_counts[bowl_counts['Overs'] > 20].sort_values('Economy').head(10)
    fig_econ = px.bar(best_econ, x='bowler', y='Economy', title='Best Economy Rates (>20 overs)', text_auto='.2f')
    st.plotly_chart(fig_econ, use_container_width=True)
    
    
    
    
    

#tab2 - toss vs match win alys


with tab2:
    st.header("Does Winning the Toss Matter?")

    toss_win = matches_f[matches_f['toss_winner'] == matches_f['winner']].shape[0]
    toss_total = matches_f.shape[0]
    percent = round((toss_win / toss_total) * 100, 2)
    st.subheader(f"In the selected seasons, the toss winner also won {percent}% of matches.")
    

    toss_df = matches_f.assign(Result = matches_f['toss_winner'] == matches_f['winner'])
    fig = px.histogram(toss_df, x='season', color='Result', barmode='group',
                       title='Toss Winner = Match Winner (Season‑wise)')
    st.plotly_chart(fig, use_container_width=True)
    
    
    

#head to head - tab3
with tab3:
    st.header("Head to Head Analysis")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams, index=teams.index(selected_team))
    with col2:
        team_b = st.selectbox("Team B", teams, index=(teams.index(selected_team)+1) % len(teams))

    h2h = matches_f[((matches_f['team1'] == team_a) & (matches_f['team2'] == team_b)) |
                    ((matches_f['team1'] == team_b) & (matches_f['team2'] == team_a))]

    wins_a = (h2h['winner'] == team_a).sum()
    wins_b = (h2h['winner'] == team_b).sum()
    draws  = h2h[h2h['winner'].isna()].shape[0]
    
    


    st.write(f"{team_a} vs {team_b} Results")
    fig_h2h = px.bar(x=[team_a, team_b, 'No Result'], y=[wins_a, wins_b, draws],
                     labels={'x':'Result', 'y':'Matches'}, text=[wins_a, wins_b, draws],
                     title='Head‑to‑Head Win Count')
    st.plotly_chart(fig_h2h, use_container_width=True)
    
    
    
    
    

#tab4 - player of the match
with tab4:
    st.header("Player of the Match!")

    pom = (matches_f.groupby('player_of_match').size()
                    .sort_values(ascending=False).head(15).reset_index(name='Awards'))
    fig_pom = px.bar(pom, x='player_of_match', y='Awards', title='Most Player of Match Awards', text='Awards')
    st.plotly_chart(fig_pom, use_container_width=True)
    
    
    

#tab5 - preview
with tab5:
    st.subheader("Matches.csv (filtered)")
    st.dataframe(matches_f.head(200))

    st.subheader("Deliveries.csv (filtered)")
    st.dataframe(deliveries_f.head(200))
    
    
    

#tab6- match pred
with tab6:
    st.header("🔮 Match Winner Predictor")

    try:
        with open('model/prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)

        team_list = sorted(matches['team1'].dropna().unique())
        venue_list = sorted(matches['venue'].dropna().unique())

        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Team 1", team_list)
        with col2:
            team2 = st.selectbox("Team 2", [team for team in team_list if team != team1])

        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        venue = st.selectbox("Venue", venue_list)

        if st.button("Predict Winner"):
            input_df = pd.DataFrame([[toss_winner, team1, team2, venue]], 
                                    columns=['toss_winner', 'team1', 'team2', 'venue'])

            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

            prediction = model.predict(input_encoded)[0]
            st.success(f"🏆 Predicted Match Winner: {prediction}")
    except FileNotFoundError:
        st.error("error")
