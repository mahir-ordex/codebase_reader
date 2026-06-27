import json
from app.main import build_collection

if __name__ == '__main__':
    result = build_collection('https://github.com/mahir-ordex/blog.git')
    print(json.dumps(result, indent=2))
