from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd

#For simplicity and efficiency, I will consider start years from 2013 and on. End year will be 2023.
def create_CSV_data(start_year, end_year):
    stats = pd.DataFrame()
    years = []

    # iterates through years specified
    for year in range(start_year, end_year+1):
        years.append(year)
    for year in years:
        stats = pd.concat([stats,scrape(year)], ignore_index=True)

    #Next seasons PPR score attached to each instance, drop any instances without a next year except the 2023 year we are predicting
    #Sort the DataFrame by Player and Year
    stats = stats.sort_values(by=['Player', 'Year'])
    #Create a new column for next year's PPR
    stats['NextPPR'] = stats.groupby('Player')['PPR'].shift(-1)
    #Delete any NaN except for the ones for 2023
    stats = stats[(stats['Year']==2023) | (stats['NextPPR'].notna())]
    stats = clean_df(stats)
    #raw csv file for stats from 2013 to 2023
    stats.to_csv(f'{start_year}to{end_year}stats.csv', index=False)
    
    return stats


def scrape(year):
    url = "https://www.pro-football-reference.com/years/{}/fantasy.htm#".format(year)
    html = urlopen(url)
    soup = BeautifulSoup(html, features='html.parser')

    headers = [th.text for th in soup.findAll('tr')[1].findAll('th')]
    headers = headers[1:]
    #print(headers[:5])
    
    rows = soup.findAll('tr', class_ = lambda table_rows: table_rows != "thead")
    player_stats = [[td.text for td in rows[i].findAll('td')] for i in range(len(rows))]
    player_stats = player_stats[2:]
    #print(player_stats[:5])

    stats = pd.DataFrame(player_stats, columns = headers)

    #Replace empty strings with 0. Specifically for Y/A, 2PM, and 2PP. Remove random symbols
    stats = stats.replace('', value=0)
    stats['Player'] = stats['Player'].str.replace('+', '')
    stats['Player'] = stats['Player'].str.replace('*', '')
    stats['Year'] = year

    #Adding a stat for total games in a season.
    if year >= 2021:
        stats['TG'] = 17
    else:
        stats['TG'] = 16    

    stats = rename_cols(stats)
    return stats

#Need to rename some of the repeated columns
def rename_cols(stats):
    cols = []
    yds_num = 0
    att_num = 0
    td_num = 0
    
    for column in stats.columns:
        if column == 'Yds':
            if yds_num == 0:
                cols.append('PassYds')
            elif yds_num == 1:
                cols.append('RushYds')
            elif yds_num == 2:
                cols.append('RecYds')
            yds_num += 1
        elif column == 'Att':
            if att_num == 0:
                cols.append('PassAtt')
            elif att_num == 1:
                cols.append('RushAtt')
            att_num += 1
        elif column == 'TD':
            if td_num == 0:
                cols.append('PassTD')
            elif td_num == 1:
                cols.append('RushTD')
            elif td_num == 2:
                cols.append('RecTD')
            elif td_num == 3:
                cols.append('TotTD')
            td_num += 1
        else:
            cols.append(column)
    
    stats.columns = cols
    return stats

#Function to remove unneeded stats
def clean_df(stats):
    columnsToRemove = ['Tm', 'GS', 'Y/A', 'Y/R', 'Fmb', 'FL', '2PM', '2PP', 'FantPt', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank']
        
    column_order = ['FantPos', 'Player', 'Year', 'Age', 'G', 'TG', 'Cmp', 'PassAtt', 'PassYds', 'PassTD', 'Int',
                     'RushAtt', 'RushYds', 'RushTD', 'Tgt', 'Rec',
                     'RecYds', 'RecTD', 'TotTD', 'PPR', 'NextPPR']

    stats = stats.drop(columns = columnsToRemove)

    #stats = stats.reset_index(drop=True, inplace=True)
    stats = stats.reindex(columns= column_order)

    return stats
    
create_CSV_data(2013, 2023)