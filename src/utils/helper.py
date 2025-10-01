from typing import List

def get_last_three_years_yq(current_yq: List[int]) -> List[List[int]]:
    # last three years year-quarter list
    # if the current yq is 2020, 2, then the last three years year-quarter list is [2017, 2], [2017, 3], [2017, 4], [2018, 1], [2018, 2], [2018, 3], [2018, 4], [2019, 1], [2019, 2], [2019, 3], [2019, 4], [2020, 1]
    last_three_years_yq = []
    for year in range(current_yq[0] - 3, current_yq[0] + 1):
        if year == current_yq[0] - 3:
            start_q = current_yq[1]
        else:
            start_q = 1
        if year == current_yq[0]:
            end_q = current_yq[1] - 1
        else:
            end_q = 4
        for quarter in range(start_q, end_q + 1):
            last_three_years_yq.append([year, quarter])

    return last_three_years_yq