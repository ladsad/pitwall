import sys
import re
import subprocess
from datetime import datetime, timezone
import fastf1

def update_config(season, event, round_number):
    with open('config.py', 'r') as f:
        content = f.read()
    
    content = re.sub(r'SEASON\s*=\s*\d+', f'SEASON = {season}', content)
    content = re.sub(r'EVENT\s*=\s*".*"', f'EVENT = "{event}"', content)
    content = re.sub(r'ROUND_NUMBER\s*=\s*\d+', f'ROUND_NUMBER = {round_number}', content)
    
    with open('config.py', 'w') as f:
        f.write(content)

def main():
    season = 2026
    print(f"Fetching schedule for season {season}...")
    schedule = fastf1.get_event_schedule(season)
    
    now = datetime.now(timezone.utc)
    
    events_run = 0
    for _, row in schedule.iterrows():
        try:
            event_date = row['EventDate']
            if event_date.tzinfo is None:
                event_date = event_date.tz_localize('UTC')
        except Exception:
            event_date = row['EventDate']
            
        if event_date < now and row['EventFormat'] != 'testing':
            round_number = row['RoundNumber']
            event_name = row['EventName']
            
            print(f"\n{'='*60}")
            print(f"--- Predicting for Round {round_number}: {event_name} ---")
            print(f"{'='*60}")
            
            update_config(season, event_name, round_number)
            
            env = dict(sys.modules['os'].environ)
            env['PYTHONIOENCODING'] = 'utf-8'
            
            try:
                subprocess.run([sys.executable, 'notebooks/06_predict.py'], env=env, check=True)
                events_run += 1
            except subprocess.CalledProcessError as e:
                print(f"Failed to run predictions for {event_name}. Error: {e}")
                continue

    print(f"\nFinished processing predictions for {events_run} events.")

if __name__ == '__main__':
    main()
