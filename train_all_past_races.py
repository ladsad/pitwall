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
        # EventDate is usually the race date
        # fastf1 EventDate might be naive or tz-aware depending on version, let's ensure it's comparable
        try:
            event_date = row['EventDate']
            if event_date.tzinfo is None:
                event_date = event_date.tz_localize('UTC')
        except Exception:
            event_date = row['EventDate']
            
        # We process races that have already happened (or are currently happening)
        # Skip testing events
        if event_date < now and row['EventFormat'] != 'testing':
            round_number = row['RoundNumber']
            event_name = row['EventName']
            
            print(f"\n{'='*60}")
            print(f"--- Training for Round {round_number}: {event_name} ---")
            print(f"{'='*60}")
            
            update_config(season, event_name, round_number)
            
            # Run the pipeline
            env = dict(sys.modules['os'].environ)
            env['PIPELINE'] = 'weekend_rf'
            
            try:
                subprocess.run([sys.executable, 'run_pipeline.py'], env=env, check=True)
                events_run += 1
            except subprocess.CalledProcessError as e:
                print(f"Failed to run pipeline for {event_name}. Error: {e}")
                # We can choose to break or continue; let's continue to the next available race
                continue

    print(f"\nFinished processing {events_run} events.")

if __name__ == '__main__':
    main()
