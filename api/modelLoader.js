const { spawn } = require('child_process');

const predict = (data) => {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn(
      'python',
      ['src/model_scripts/predict_data.py', JSON.stringify(data)],
      { cwd: '/Users/lionelamuzu/Desktop/lionel_sekyie/petfinder' }
    );

    let scriptOutput = '';
    pythonProcess.stdout.on('data', (data) => {
      scriptOutput += data.toString();
    });

    let scriptError = '';
    pythonProcess.stderr.on('data', (data) => {
      scriptError += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python script exited with code ${code}`);
        reject(scriptError);
      } else {
        resolve(scriptOutput);
      }
    });
  });
};

module.exports = { predict };
