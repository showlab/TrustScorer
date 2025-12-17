import os

# set your vpn proxy in China
print("if you are in China, need import proxy in explore_utils")
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10809"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10809"
