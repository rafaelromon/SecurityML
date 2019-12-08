#! usr/bin/env python3


import shutil

import praw
# import progressbar
import requests

import nsfw_classification.utils.settings as settings

LIMIT = 800

reddit = praw.Reddit(client_id=settings.ID,
                     client_secret=settings.SECRET,
                     user_agent=settings.USER_AGENT,
                     username=settings.USERNAME,
                     password=settings.PASSWORD)


def scrape_subreddit(name, type):
    subreddit = reddit.subreddit(name)

    print("www.reddit.com/r/%s" % name)

    count = 0
    stage = "train"

    print("SCRAPPING TRAINING SET:")

    limit = LIMIT + LIMIT * 0.25

    # bar = progressbar.ProgressBar(max_value=LIMIT)

    for submission in subreddit.top(limit=limit):
        resp = requests.get(submission.url, stream=True)
        resp.raw.decode_content = True

        filename = submission.url.rsplit('/', 1)[-1]

        if ".jpg" in filename or ".png" in filename:
            local_file = open("../dataset/%s/%s/%s" % (stage, type, filename), "wb")
            shutil.copyfileobj(resp.raw, local_file)
            del resp

        count = count + 1

        # bar.update(count)

        if count >= LIMIT:
            print("SCRAPPING VALIDATION SET:")

            count = 0
            # bar = progressbar.ProgressBar(max_value=LIMIT * 0.25)

            stage = "validation"


if __name__ == '__main__':
    scrape_subreddit("gonewild", "nsfw")
    scrape_subreddit("selfies", "sfw")
