const modelLoader = require('./modelLoader');

exports.makePrediction = async (req, res) => {
  try {
    const data = req.body;
    const prediction = await modelLoader.predict(data);
    res.json({ prediction });
  } catch (error) {
    res.status(500).send(error.message);
  }
};
