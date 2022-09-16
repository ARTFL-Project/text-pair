module.exports = {
    devServer: {
        compress: true,
        allowedHosts: 'all',
        headers: {
            'Access-Control-Allow-Origin': '*'
        }
    },
    // chainWebpack: (config) => {
    //     config.resolve.alias.set('vue', '@vue/compat')
    //     config.module
    //         .rule('vue')
    //         .use('vue-loader')
    //         .tap((options) => {
    //             return {
    //                 ...options,
    //                 compilerOptions: {
    //                     compatConfig: {
    //                         MODE: 2
    //                     }
    //                 }
    //             }
    //         })
    // },
    publicPath: process.env.NODE_ENV === 'production' ?
        getAppPath() : '/'
}


function getAppPath() {
    const globalConfig = require('./appConfig.json')
    return globalConfig.appPath
}