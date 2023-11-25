// const { spawn } = require('child_process');

// const predict = (data) => {
//   return new Promise((resolve, reject) => {
//     const pythonProcess = spawn(
//       'python',
//       ['src/model_scripts/predict_data.py', JSON.stringify(data)],
//       { cwd: '/Users/lionelamuzu/Desktop/lionel_sekyie/petfinder' }
//     );

//     pythonProcess.stdout.on('data', (data) => {
//       resolve(data.toString());
//     });

//     pythonProcess.stderr.on('data', (data) => {
//       reject(data.toString());
//     });
//   });
// };

// module.exports = { predict };

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

    pythonProcess.stdout.on('end', () => {
      resolve(scriptOutput);
    });

    pythonProcess.stderr.on('data', (data) => {
      reject(data.toString());
    });
  });
};

module.exports = { predict };
