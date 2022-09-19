module.exports = {
    devServer: {
        compress: true,
        allowedHosts: 'all',
    },
    publicPath: process.env.NODE_ENV === 'production' ?
        getAppPath() : '/'
}


function getAppPath() {
    const globalConfig = require('./appConfig.json')
    return globalConfig.appPath
}