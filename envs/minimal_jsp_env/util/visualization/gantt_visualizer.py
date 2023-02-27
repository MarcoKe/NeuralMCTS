import datetime
import pandas as pd
import plotly.express as px

JOB_STR = 'job_id'
OPERATION_STR = 'operation_id'
MACHINE_STR = 'machine_id'
START_STR = "start_time"
START_DATE_STR = "start_datetime"
END_STR = "finish_time"
END_DATE_STR = "end_datetime"

UNIT = 'minutes'

def convert(data):
    new_data = []

    for job in data:
        for (op, start, end) in job:
            row = [op.job_id, op.op_id, op.machine_type, start, end]
            new_data.append(row)

    return new_data

def create_gantt(data, start_datetime=datetime.datetime(year=2022, month=1, day=1, hour=8), title='gantt'):
    data = convert(data)
    df = pd.DataFrame(data, columns=[JOB_STR, OPERATION_STR, MACHINE_STR, START_STR, END_STR])

    df[START_STR] = df[START_STR].apply(lambda x: start_datetime + pd.Timedelta(x, unit=UNIT))
    df[START_DATE_STR] = df[START_STR].apply(lambda start: pd.to_datetime(str(start)))
    df[END_STR] = df[END_STR].apply(lambda x: start_datetime + pd.Timedelta(x, unit=UNIT))
    df[END_DATE_STR] = df[END_STR].apply(lambda end: pd.to_datetime(str(end)))

    fig = px.timeline(df, x_start=START_DATE_STR, x_end=END_DATE_STR, y=MACHINE_STR, color=JOB_STR, opacity=0.5,
                      title=title)

    # fig.write_html(f'{title}.html')
    fig.show(renderer="browser")
    return fig
