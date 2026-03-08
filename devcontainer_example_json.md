{
  "name": "cuda-python-dev",

  "build": {
      "dockerfile": "../Dockerfile",
      "context": ".."
  },

  "runArgs": [
      "--gpus",
      "all"
  ],

  "workspaceFolder": "/workspace",

  "customizations": {
      "vscode": {
          "extensions": [
              "ms-python.python",
              "ms-python.vscode-pylance",
              "ms-toolsai.jupyter",
              "ms-vscode.cpptools"
          ]
      }
  },

  "postCreateCommand": "python3 -m pip install --upgrade pip",

  "remoteUser": "root"
}