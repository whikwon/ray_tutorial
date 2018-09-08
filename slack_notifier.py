import os
from slackclient import SlackClient


token = os.environ.get('SLACK_TOKEN')

def slack_message(message, channel, token):
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel,
                text=message, username='Alarm',
                icon_emoji=':robot_face:')


if __name__ == '__main__':
    slack_message("Hello, You got error.", "notify", token)
