module.exports = {
	devServer: {
		disableHostCheck: true,
		host: "0.0.0.0",
		headers: {
			'Access-Control-Allow-Origin': '*'
		}
	},
	configureWebpack: {
		output: {
			globalObject: 'this'
		}
	},
	baseUrl: process.env.NODE_ENV === 'production'
    ? getAppPath()
    : '/'
}


function getAppPath() {
	const globalConfig = require('./appConfig.json')
	return globalConfig.appPath
}