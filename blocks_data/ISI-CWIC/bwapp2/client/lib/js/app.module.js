/*!
 * 
 * Angle - Bootstrap Admin App + AngularJS
 * 
 * Version: 3.1.0
 * Author: @themicon_co
 * Website: http://themicon.co
 * License: https://wrapbootstrap.com/help/licenses
 * 
 */

// APP START
// ----------------------------------- 

(function() {
    'use strict';

    angular
        .module('angle', [
            'app.core',
            'app.routes',
            'app.sidebar',
            'app.navsearch',
            //'app.preloader',
            //'app.loadingbar',
            'app.translate',
            'app.settings',
            'app.utils',
            'app.generate',
            'app.togglestate'
        ]);
    
    if (Meteor.isCordova)
      angular.element(document).on("deviceready", onReady);
    else
      angular.element(document).ready(onReady);
    
    function onReady() {
      angular.bootstrap(document, ['angle']);
    }

})();

