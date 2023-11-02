import numpy as np
import pandas as pd
import swifter

class VotingData():
    """
    The PrefList class is an object that will hold all the data of a certain election.
    """
    def __init__(self, filepath, num_ranks):
        self.filepath = filepath
        self.num_ranks = num_ranks
        
        ranks = []
        for i in range(1, num_ranks + 1):
            ranks.append('rank'+str(i))
            
        self.ranks = ranks

        df = pd.read_csv(filepath, usecols=ranks)

        def _left_adj_and_remove_dupes(row, n=5):
            """
            Shifts values in a row of n columns to the left so all '' appear to the right.
            """
            seen = []
            for i in range(n):
                if row.iloc[i] in seen:
                    row.iloc[i] = ''
                if row.iloc[i] == '' and i < n - 1:
                    j = i + 1
                    swapped = False
                    while j <= n-1 and not swapped:
                        if row.iloc[j] in seen:
                            row.iloc[j] = ''
                        if row.iloc[j] != '':
                            row.iloc[i], row.iloc[j] = row.iloc[j], row.iloc[i]
                            swapped = True
                        j += 1
                    if not swapped:
                        return row
                seen.append(row.iloc[i])
            return row

        df = df.groupby(ranks, as_index=False).size()
        df = df[(df != 'overvote').all(axis=1)]
        df = df.replace('skipped', '')
        df = df.groupby(ranks, as_index=False).sum()
        df = df.swifter.apply(_left_adj_and_remove_dupes, axis = 1, args=(num_ranks,))
        df = df.groupby(ranks, as_index=False).sum()
        df = df.loc[df['rank1'] != '']
        df = df.rename(columns={'size':'count'})
        
        self.ballots = df.sort_values('count', ascending=False)
        self.candidates = tuple([x for x in pd.unique(self.ballots[ranks].to_numpy().ravel()) if x != ''])
        
    def _remove_and_left_adj(self, row, n, removed):
        """
        input: row is a row in a ballot. It is assumed that all none '' entries come first and are unique.
            n is the number of ranks in the ballot. 
            removed is the candidate to be removed

        output: a row with removed replaced by '' and all entries left adjusted.
        """
        i = 0
        found = False
        while i < n:
            if row.iloc[i] == removed:
                found = True
                break
            i += 1

        if found:
            while i < n and row.iloc[i] != '':
                if i == n - 1:
                    row.iloc[i] = ''
                else:
                    row.iloc[i] = row.iloc[i+1]
                i += 1
        return row
    
    def plurality_winner(self):
        df = self.ballots[['rank1','count']].groupby('rank1').sum().sort_values('count',ascending=False)
        num_votes = df['count'].sum()
        df['percent'] = round(100*df['count']/num_votes, 1)
        return df
    
    def instant_runoff_winner(self):
            df = self.ballots.copy()
            winner =  False
            result = pd.DataFrame({}, index=self.candidates) 
            rnd = 1
            while True:
                curr_rnd = df[['rank1','count']].groupby('rank1').sum().sort_values('count',ascending=False)
                num_votes = curr_rnd['count'].sum()
                curr_rnd['percent'] = round(100*curr_rnd['count']/num_votes, 1)
                curr_rnd = curr_rnd.rename(columns={'count':'count '+str(rnd),'percent':'percent '+str(rnd)})

                # append curr_rnd to results
                result = pd.concat([result,curr_rnd], axis=1)

                if curr_rnd.max()['percent '+str(rnd)] > 50:
                    result = result.sort_values('count 1', ascending=False)
                    return result
                else:
                    loser = curr_rnd.idxmin()['count '+str(rnd)]
                    df[(df == loser).any(axis=1)] = df[(df == loser).any(axis=1)].swifter.progress_bar(False).apply(self._remove_and_left_adj, axis=1, args=(self.num_ranks,loser))
                    df = df.groupby(self.ranks, as_index=False).sum()
                    df = df.loc[df['rank1'] != '']
                rnd += 1
                
    def compare(self, can1, can2):
        others = [x for x in self.candidates if x not in [can1,can2]]
        df = self.ballots.replace(others, '')
        df = df.groupby(self.ranks, as_index=False).sum()
        df = df[(df == can1).any(axis=1) | (df == can2).any(axis=1)]

        def _row_winner(row, n):
            for i in range(n):
                if row.iloc[i] != '':
                    return row.iloc[i]

        df['Candidate'] = df.apply(_row_winner, axis=1, args=(self.num_ranks,))
        df = df[['Candidate','count']].groupby('Candidate').sum()
        return df
        
    def pairwise_winners(self):
        """
        returns a matrix showing the one on one results of row versus column.
        """
        res = pd.DataFrame(0,columns = self.candidates, index=self.candidates)

        for i, can1 in enumerate(self.candidates[:-1]):
            for _, can2 in enumerate(self.candidates[i+1:]):
                df = self.compare(can1, can2)
                a = df.loc[can1, 'count']
                b = df.loc[can2, 'count']
                res.loc[can1, can2] = round(100*a/(a+b),1)
                res.loc[can2, can1] = 100 - res.loc[can1, can2]

        return res
    
    def condorcet_winner(self):
        """Returns the candidate that beats all others in pairwise elections. If no such
        candidate exists, returns "No Condorcet candidate".

        We search for this candidate by following 
        """
        possible_winners = list(self.candidates)

        while possible_winners:
            cand = possible_winners[0]
            for rival in [x for x in self.candidates if x != cand]:
                df = self.compare(cand, rival)
                if df.loc[cand, 'count'] <= df.loc[rival, 'count']:
                    possible_winners.remove(cand)
                    break
                else:
                    if rival in possible_winners:
                        possible_winners.remove(rival)
            if cand in possible_winners: 
                return cand

        return "No Condorcet candidate"
    
    def voting_results(self):
        plurw = self.plurality_winner()['count'].idxmax()
        
        irw = self.instant_runoff_winner().iloc[:,-1].idxmax()
        
        condw = self.condorcet_winner()
        
        res = {'Plurality': plurw, "Instant Runoff": irw, "Condorcet Candidate": condw}
        
        return res
        
    def STV_winners(self, n):
        """
        input: n is the number of candidates being elected.
        
        output: The results of the single transferable vote algorithm.
        """
        pass