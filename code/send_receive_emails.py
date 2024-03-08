'''send-receive-emails by Samuel Collier & Lucia Gordon'''

# imports
import imaplib
import numpy as np
import os
import pandas as pd
from datetime import datetime
from email import encoders, message_from_bytes, policy
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from smtplib import SMTP_SSL
from ssl import create_default_context

class SendReceiveEmail:
    def __init__(self, folder, batch_number):
        with open('code/send-receive-emails.config') as config:
            credentials = {line.strip().split('|')[0]: line.strip().split('|')[1] for line in config.readlines()}
    
        self.sender_email = credentials.get('sender_email')
        self.sender_password = credentials.get('sender_password')
        self.receiver_email = credentials.get('receiver_email')
        self.attachment_path = f'{folder}/data/batch-{batch_number}-shapefile.zip'
        self.batch_number = batch_number
        self.batch_labels = {}

        self.send_email()
        self.read_email()

    def send_email(self):
        email_message = MIMEMultipart()
        email_message.add_header('To', self.receiver_email)
        email_message.add_header('From', self.sender_email)
        email_message.add_header('Subject', f'Batch {self.batch_number} Ready for Labeling')

        text = 'Please open the attached shapefile to verify whether the boxed regions contain a midden. Fill in the "label" column by entering "1" if the box corresponding to the identifier contains a midden and "0" otherwise. Then attach the filled-in CSV file as a reply to this email.\n'
        email_text = MIMEText(text, 'plain')
        email_message.attach(email_text)
        attachment = MIMEBase('application', 'zip')
        attachment.set_payload(open(self.attachment_path, 'rb').read())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', 'attachment', filename=f'batch-{self.batch_number}')
        email_message.attach(attachment)
        context = create_default_context()

        with SMTP_SSL('smtp.gmail.com', port = 465, context = context) as smtp_server:
            smtp_server.login(self.sender_email, self.sender_password)
            smtp_server.sendmail(self.sender_email, self.receiver_email, email_message.as_string())
            smtp_server.quit()

        print('Email sent')

    def read_email(self):
        imap_server = imaplib.IMAP4_SSL(host = 'imap.gmail.com')
        imap_server.login(self.sender_email, self.sender_password)
        imap_server.select('INBOX')
        all_message_numbers = []
        
        while not all_message_numbers: # wait until reply comes in
            imap_server.noop() # refresh
            response_code, message_numbers = imap_server.search(None, f'HEADER Subject "RE: Batch {self.batch_number} Ready for Labeling" UNSEEN') # search for emails with this subject
            all_message_numbers += message_numbers[0].split()

        response_code, message_data = imap_server.fetch(all_message_numbers[-1], '(RFC822)')

        for response in message_data:
            if isinstance(response, tuple):
                msg = message_from_bytes(response[1], policy = policy.default) # parse a bytes email into a message object

                for part in msg.walk():
                    if part.get_content_type() == 'text/csv':
                        open('batch-labels-returned.csv', 'wb').write(part.get_payload(decode = True))
                        identifiers = pd.read_csv('batch-labels-returned.csv').loc[:, 'id'].to_numpy()
                        labels = pd.read_csv('batch-labels-returned.csv').loc[:, 'label'].to_numpy()
                        self.batch_labels = dict(zip(identifiers, labels))
                        os.remove('batch-labels-returned.csv')
                
        imap_server.close()
        imap_server.logout()
