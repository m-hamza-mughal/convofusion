# create a function to parse textgrid files

import textgrid as tg
import numpy as np
import json

def parse_textgrid(textgrid_path):
    """Parse textgrid file and return a dictionary of the form:
    {
        "text": [str],
        "start": [float],
        "end": [float],
        "duration": [float]
    }
    """
    # breakpoint()
    textgrid = tg.TextGrid.fromFile(textgrid_path)
    text = []
    start = []
    end = []
    duration = []
    for interval in textgrid[0]:
        text.append(interval.mark)
        start.append(interval.minTime)
        end.append(interval.maxTime)
        duration.append(interval.maxTime - interval.minTime)
    return {
        "text": np.array(text),
        "start": np.array(start),
        "end": np.array(end),
        "duration": np.array(duration)
    }
    # return text


if __name__ == "__main__":
    textgrid_path = "./datasets/beat_english_v0.2.1/1/1_wayne_0_2_2.TextGrid"
    text_data = parse_textgrid(textgrid_path)
    breakpoint()
    print(text_data)
    