const controller = require('./controller');

module.exports = (app) => {
  app.post('/predict', controller.makePrediction);
  app.post('/createUser', controller.createUser);
};
