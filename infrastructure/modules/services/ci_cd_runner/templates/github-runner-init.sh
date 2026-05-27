# Create a folder
mkdir actions-runner && cd actions-runner# Download the latest runner package
curl -o actions-runner-linux-x64-2.334.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.334.0/actions-runner-linux-x64-2.334.0.tar.gz# Optional: Validate the hash
echo "048024cd2c848eb6f14d5646d56c13a4def2ae7ee3ad12122bee960c56f3d271  actions-runner-linux-x64-2.334.0.tar.gz" | shasum -a 256 -c# Extract the installer
tar xzf ./actions-runner-linux-x64-2.334.0.tar.gz

# Create the runner and start the configuration experience
./config.sh --url https://github.com/Vegamaps --token AV5XHER4UCKKQFHVFZT2M2TJ7XDI2 # token must be updated depending on the organization
nohup ./run.sh > run.log 2>&1 & # running the github actions runner in the background mode