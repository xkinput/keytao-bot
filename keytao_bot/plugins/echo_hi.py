"""
Echo plugin: reply user message + hi username
"""
from nonebot import on_message
from nonebot.adapters import Bot, Event
from nonebot.rule import to_me

# Create message handler (respond to @bot or private message)
echo_hi = on_message(rule=to_me(), priority=10, block=True)


@echo_hi.handle()
async def handle_echo_hi(bot: Bot, event: Event):
    """
    Handle message and reply with: user's message + hi + username
    """
    # Get message text
    message_text = event.get_plaintext().strip()
    
    # Get username (different for QQ and Telegram)
    try:
        # Try to get username from event
        username = event.get_user_id()
        
        # Try to get more friendly name if available
        if hasattr(event, 'sender'):
            sender = event.sender
            if hasattr(sender, 'nickname'):
                username = sender.nickname
            elif hasattr(sender, 'card') and sender.card:
                username = sender.card
            elif hasattr(sender, 'username') and sender.username:
                username = sender.username
    except Exception:
        username = "friend"
    
    # Build reply message
    if message_text:
        reply = f"{message_text} hi {username}"
    else:
        reply = f"hi {username}"
    
    await echo_hi.finish(reply)
