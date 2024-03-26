


css = '''
<style>
.chat-message {
    padding: 0.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    align-items: center;
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 5%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 95%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''
bot_template = '''
<div class="chat-message bot">
    <div class="avatar" style="font-size: 48px;"> <!-- Adjust font size as needed -->
        🤖
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar" style="font-size: 48px;"> <!-- Adjust font size as needed -->
        👨🏼
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''