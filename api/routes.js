const controller = require('./controller');

module.exports = (app) => {
  app.post('/predict', controller.makePrediction);
  // ... other routes can be added here
};
