import numpy as np
import pandas as pd

final_pred = pd.read_csv("data/final_predictions.csv")
def getMatch(team1,team2):
    data = final_pred[(final_pred['Team1']== team1) & (final_pred['Team2']==team2)]
    print(f''
          '+----------------------------------------------------+\n'
         f'|    Home Team: {team1}  |  Away Team:{team2}         |\n'
         f'| Home Win : {np.round(1/data['HomeWin%'].values[0],2)}  || Away Win : {np.round(1/data['AwayWin%'].values[0],2)} || Draw : {np.round(1/data['Draw%'].values[0],2)}  |\n'
          '+----------------------------------------------------+')
def getTeam(team):
    pass

if __name__ == '__main__':
    getMatch('Man City','West Ham')