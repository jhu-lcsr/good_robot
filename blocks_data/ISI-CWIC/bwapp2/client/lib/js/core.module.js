(function() {
  'use strict';

  angular
    .module('app.core', [
      'angular-meteor',
      'angular-meteor.auth',
      'ngRoute',
      'ngAnimate',
      'ngStorage',
      'ngCookies',
      'pascalprecht.translate',
      'ui.bootstrap',
      'ui.router',
      'oc.lazyLoad',
      'cfp.loadingBar',
      'ngSanitize',
      'ngResource',
      //'ui.utils'
    ]);
})();