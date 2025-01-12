"""
Data loader for Clio conversation data from Excel files.
"""
import pandas as pd
from datetime import datetime
import re
import uuid
from pathlib import Path
import logging
from openclio import Conversation, ConversationTurn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_valid_row(row: pd.Series) -> tuple[bool, str]:
    """Return (is_valid, error_message) for a row"""
    # Check required fields exist
    for field in ['Prompt', 'Response', 'Country', 'Email', 'SessionDate', 'id']:
        if pd.isna(row[field]):
            return False, f"Missing {field}"
            
    # Basic format checks
    if not re.match(r'^[A-Z]{2}$', str(row['Country'])):
        return False, f"Invalid country code: {row['Country']}"
        
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', str(row['Email'])):
        return False, f"Invalid email: {row['Email']}"
        
    # Validate UUID
    try:
        uuid.UUID(str(row['id']))
    except ValueError:
        return False, f"Invalid UUID: {row['id']}"
        
    # Validate date - accept datetime objects or common string formats
    if isinstance(row['SessionDate'], (pd.Timestamp, datetime)):
        return True, ""
        
    date_formats = [
        '%d/%m/%Y %I:%M:%S %p', # 24/10/2024 9:09:34 am
        '%Y-%m-%d %H:%M:%S.%f', # 2024-09-26 02:01:55.753000
        '%Y-%m-%d %H:%M:%S',    # 2024-09-26 02:01:55
    ]
    
    for fmt in date_formats:
        try:
            datetime.strptime(str(row['SessionDate']), fmt)
            return True, ""
        except ValueError:
            continue
            
    return False, f"Invalid date format: {row['SessionDate']}"

def load_conversation_data(file_path: Path, ignore_errors: bool = False) -> list[Conversation]:
    """Load conversation data from Excel file"""
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}")
    
    # Check required columns exist
    required_cols = {'Prompt', 'Response', 'Country', 'Email', 'SessionDate', 'id'}
    if missing := required_cols - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")
    
    # Filter valid rows
    valid_rows = []
    for idx, row in df.iterrows():
        is_valid, error = is_valid_row(row)
        if is_valid:
            valid_rows.append(idx)
        elif not ignore_errors:
            raise ValueError(f"Row {idx + 2}: {error}")
        else:
            logger.warning(f"Skipping row {idx + 2}: {error}")
    
    df = df.iloc[valid_rows]
    
    # Build conversations (oldest messages last)
    convs = {}
    for _, row in df.iloc[::-1].iterrows():
        if row['id'] not in convs:
            convs[row['id']] = Conversation(
                id=row['id'],
                metadata={
                    'country': row['Country'],
                    'user_id': row['Email'],
                    'session_date': row['SessionDate']
                },
                turns=[]
            )
        
        convs[row['id']].turns.extend([
            ConversationTurn(role="user", content=row['Prompt']),
            ConversationTurn(role="assistant", content=row['Response'])
        ])
    
    conversations = list(convs.values())
    logger.info(f"Loaded {len(conversations)} conversations")
    return conversations

if __name__ == "__main__":
    try:
        convs = load_conversation_data(Path("conversations.xlsx"), ignore_errors=True)
        
        if convs:
            # Print example conversation
            conv = convs[0]
            print(f"\nExample Conversation:")
            print(f"ID: {conv.id}")
            print(f"Country: {conv.metadata['country']}")
            print(f"Email: {conv.metadata['user_id']}")
            print(f"Date: {conv.metadata['session_date']}")
            for turn in conv.turns:
                print(f"\n{turn.role}: {turn.content}")
                
    except Exception as e:
        logger.error(f"Error: {e}") 