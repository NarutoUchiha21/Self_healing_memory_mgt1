@echo off
echo Running Memory Manager with admin privileges...
powershell -Command "Start-Process -FilePath python -ArgumentList 'd:\clg\COA\Self_healing_memory\app\healer_agent.py ' -Verb RunAs"
exit
