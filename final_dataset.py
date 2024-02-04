import copy
import random

import numpy as np
import pandas as pd

ROUND = 6


filenames = ["data/2016-17.csv",
             "data/2017-18.csv",
             "data/2018-19.csv",
             "data/2019-20.csv",
             "data/2020-21.csv",
             "data/2021-22.csv",
             "data/2022-23.csv"]
season1 = pd.read_csv(filenames[0])
season2 = pd.read_csv(filenames[1])
season3 = pd.read_csv(filenames[2])
season4 = pd.read_csv(filenames[3])
season5 = pd.read_csv(filenames[4])
season6 = pd.read_csv(filenames[5])
season7 = pd.read_csv(filenames[6])

seasons = [season1,
           season2,
           season3,
           season4,
           season5,
           season6,
           season7]

current_season = pd.read_csv('data/2023-24.csv')
current_teams = current_season['Home Team']
current_teams = current_teams.drop_duplicates()
current_teams = current_teams.values.tolist()

final_data = pd.DataFrame(columns=['MatchRating','NumofHomeWins','NumofDraws','NumofAwayWins', 'HomeWin%',
                                   'Draw%','AwayWin%'])
# final_data.columns = header_line

copy_teams = copy.deepcopy(current_teams)


def get_goals(team, season, rounds_to_look):
    scored = 0
    conceded = 0
    form = 0
    global ROUND
    home_matches = season[season['Home Team'] == team]
    away_matches = season[season['Away Team'] == team]
    all_matches = pd.concat([home_matches, away_matches])
    # all_matches = all_matches.sort_values(by = lambda x : x['Round Number'], ascending= True)
    for m in range(len(all_matches)):
        match = all_matches.iloc[m]
        round = int(match['Round Number'])
        if rounds_to_look[0] < round < rounds_to_look[1]:
            if match['Home Team'] == team:
                scored += int(match['Result'].split("-")[0].strip())
                conceded += int(match['Result'].split("-")[1].strip())
            else:
                scored += int(match['Result'].split("-")[1].strip())
                conceded += int(match['Result'].split("-")[0].strip())

    return scored, conceded


def get_goal_difference(team, curr_round, season):
    global ROUND
    if curr_round > 21 and season.iloc[0]['Away Team'] == 'Man City':
        rounds_to_look = (21-ROUND,ROUND)
        scored, conceded = get_goals(team, season, rounds_to_look)
        return scored - conceded
    if curr_round < 6:
        last_season = pd.read_csv("./data/2022-23.csv")
        rounds_to_look = (38 - ROUND, 38 )
        scored, conceded = get_goals(team, last_season, rounds_to_look)
        return scored - conceded

'''
for s in range(len(seasons)):
    season = seasons[s]
    for idx,match in season.iterrows():
        final = match
        team1 = match ['Home Team']
        team2 = match ['Away Team']
        round_number = match ['Round Number']
        home_gd= get_goal_difference(team1, round_number, season)
        away_gd= get_goal_difference(team2, round_number, season)
        match_rating = home_gd - away_gd
        if match_rating < -30 or match_rating >30:
            continue

        exists = match_rating in final_data['MatchRating'].values
        if not exists:
            new_row = {'MatchRating': match_rating, 'NumofHomeWins': 0, 'NumofAwayWins': 0, 'NumofDraws': 0, 'HomeWin%': 0, 'AwayWin%': 0, 'Draw%' : 0}
            final_data.loc[len(final_data)] = new_row

        num_of_home = final_data.loc[final_data['MatchRating'] == match_rating, 'NumofHomeWins'].values[0]
        num_of_away = final_data.loc[final_data['MatchRating']== match_rating, 'NumofAwayWins'].values[0]
        num_of_draws = final_data.loc[final_data['MatchRating'] == match_rating, 'NumofDraws'].values[0]

        print(final['Result'])
        if int(final['Result'].split("-")[0].strip()) > int(final['Result'].split("-")[1].strip()):
            final_data.loc[final_data['MatchRating'] == match_rating, 'NumofHomeWins'] += 1
            num_of_home = final_data.loc[final_data['MatchRating'] == match_rating, 'NumofHomeWins'].values[0]
            precentage = (num_of_home / (num_of_home + num_of_away + num_of_draws)).astype(float)
            final_data.loc[final_data['MatchRating'] == match_rating, 'HomeWin%'] = round(precentage,2)
        elif int(final['Result'].split("-")[1].strip()) > int(final['Result'].split("-")[0].strip()):
            final_data.loc[final_data['MatchRating'] == match_rating, 'NumofAwayWins'] += 1
            num_of_away = final_data.loc[final_data['MatchRating'] == match_rating, 'NumofAwayWins'].values[0]
            precentage = (num_of_away / (num_of_home + num_of_away + num_of_draws)).astype(float)
            final_data.loc[final_data['MatchRating'] == match_rating, 'AwayWin%'] = round(precentage,2)
        else:
            final_data.loc[final_data['MatchRating'] == match_rating, 'NumofDraws'] += 1
            num_of_draws = final_data.loc[final_data['MatchRating'] == match_rating, 'NumofDraws'].values[0]
            precentage = (num_of_draws / (num_of_home + num_of_away + num_of_draws)).astype(float)
            final_data.loc[final_data['MatchRating'] == match_rating, 'Draw%'] = round(precentage,2)

final_data = final_data.sort_values(by = ['MatchRating'],ascending=True)
final_data.to_csv('data/final_data.csv', index=False)
'''
