1. create & activate python env
   MacOS: python3 -m venv agent / source agent/bin/activate
   Window: python -m venv agent / agent\Scripts\activate

2. Window: trong trường hợp thư viện k có quyền truy cập (Get-ExecutionPolicy = Restricted)
Chạy: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned   


3. Install requirement packages: pip install --ignore-installed -r requirement.txt
