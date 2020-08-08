/**=========================================================
 * Module: config.js
 * App routes and resources configuration
 =========================================================*/


(function() {
  'use strict';
  angular
      .module('app.routes')
      .config(routesConfig);

  routesConfig.$inject = ['$stateProvider', '$locationProvider', '$urlRouterProvider', 'RouteHelpersProvider'];
  function routesConfig($stateProvider, $locationProvider, $urlRouterProvider, helper) {

    // Set the following to true to enable the HTML5 Mode
    // You may have to set <base> tag in index and a routing configuration in your server
    $locationProvider.html5Mode(true);

    // defaults to dashboard
    $urlRouterProvider.otherwise('/404');

    // 
    // Application Routes
    // -----------------------------------   
    $stateProvider
      .state('404', {
        url: '/404',
        title: "Not Found",
        templateUrl: helper.basepath('404.html'),
        resolve: helper.resolveFor('modernizr', 'icons'),
        controller: ["$rootScope", function ($rootScope) {
          $rootScope.app.layout.isBoxed = false;
        }]
      })
      .state('main', {
        url: '/main',
        title: "CwC ISI",
        templateUrl: helper.basepath('main.html'),
        resolve: helper.resolveFor('modernizr', 'icons'),
        onEnter: ["$rootScope", "$state", "$auth", function ($rootScope, $state, $auth) {
          $rootScope.app.layout.isBoxed = false;
          $auth.requireUser().then(function (usr) {
            if(!$rootScope.isRole(usr, 'guest')){
              if (usr) $state.go('app.genworld');
            }
          });
          Accounts.onLogin(function (user) {
            $state.go('app.root')
          })
        }]
      })
      .state('app', {
        url: '',
        abstract: true,
        templateUrl: helper.basepath('app.html'),
        resolve: helper.resolveFor('modernizr', 'icons', 'toaster')
      })
      .state('app.root', {
        url: '/',
        title: "CwC ISI",
        onEnter: ['$rootScope', '$state', '$auth', function ($rootScope, $state, $auth) {
          $auth.requireUser().then(function (usr) {
            if (usr) {
              if($rootScope.isRole(usr, 'guest')){
                $state.go('main')
              }
              else $state.go('app.genworld');
            }
            else $state.go('main');
          }, function (err) {
            $state.go('main');
          });
        }]
      })
      .state('app.genworld', {
        url: '/genworld?sid',
        title: 'Generate World',
        templateUrl: helper.basepath('genworld.html'),
        resolve: angular.extend(
          {
            "currentUser": ["$auth", "$rootScope", function ($auth, $rootScope) {
              return $auth.requireValidUser(function (user) {
                return !$rootScope.isRole(user, 'guest');
              });
            }]
          },  //simple functions appear first so data is loaded
          helper.resolveFor('babylonjs', 'ngDialog', 'ngTable')
        ),
        controller: 'genWorldCtrl'
      })
      .state('app.gensimpexp', {
        url: '/gensimpexp?sid',
        title: 'Simple Experiment',
        templateUrl: helper.basepath('gensimpexp.html'),
        resolve: angular.extend(
          {
            "currentUser": ["$auth", function ($auth) {
              return $auth.requireUser();
            }]
          }, //simple functions appear first so data is loaded
          helper.resolveFor('babylonjs', 'datatables')
        )
        ,controller: 'genSimpExpCtrl'
      })
      .state('app.gencmdexp', {
        url: '/gencmdexp?sid',
        title: 'Command Experiment',
        templateUrl: helper.basepath('gencmdexp.html'),
        resolve: angular.extend(
          {
            "currentUser": ["$auth", function ($auth) {
              return $auth.requireUser();
            }]
          }, //simple functions appear first so data is loaded
          helper.resolveFor('babylonjs', 'datatables')
        )
        ,controller: 'genCmdExpCtrl'
      })
      .state('app.gencmdjobs', {
        url: '/gencmdjobs?sid',
        title: 'Command Jobs',
        templateUrl: helper.basepath('gencmdjobs.html'),
        resolve: angular.extend(
          {
            "currentUser": ["$auth", '$rootScope', function ($auth, $rootScope) {
              return $auth.requireValidUser(function (user) {
                return !$rootScope.isRole(user, 'guest');
              });
            }]
          }, //simple functions appear first so data is loaded
          helper.resolveFor('babylonjs', 'ngDialog', 'datatables','clipboard')
        )
        ,controller: 'genCmdJobsCtrl'
      })
      .state('app.genpred', {
        url: '/genpred?sid',
        title: 'Generate Prediction',
        templateUrl: helper.basepath('genpred.html'),
        resolve: angular.extend(
          {
            "currentUser": ["$auth", '$rootScope', function ($auth, $rootScope) {
              return $auth.requireValidUser(function (user) {
                return !$rootScope.isRole(user, 'guest');
              });
            }]
          },  //simple functions appear first so data is loaded
          helper.resolveFor('babylonjs', 'ngTable')
        ),
        controller: 'genPredCtrl'
      })
      .state('app.genjobs', {
        url: '/genjobs',
        title: 'Generate Tasks View',
        templateUrl: helper.basepath('genjobs.html'),
        resolve: angular.extend(
          {
            "currentUser": ["$auth", '$rootScope', function ($auth, $rootScope) {
              return $auth.requireValidUser(function (user) {
                return !$rootScope.isRole(user, 'guest');
              });
            }]
          },  //simple functions appear first so data is loaded
          helper.resolveFor('ngDialog', 'datatables')
        ),
        controller: 'genJobsCtrl'
      })
      .state('gentask', {
        url: '/annotate?taskId&assignmentId&hitId&turkSubmitTo&workerId&report',
        title: 'Annotation Task',
        templateUrl: helper.basepath('gentask.html'),
        resolve: helper.resolveFor('modernizr', 'icons', 'toaster', 'ngDialog', 'datatables','ngDeviceDetect'),
        controller: 'genTaskCtrl'
      })
      .state('gencmdtask', {
        url: '/command?taskId&assignmentId&hitId&turkSubmitTo&workerId&report',
        title: 'Command & Response Task',
        templateUrl: helper.basepath('gencmdtask.html'),
        resolve: helper.resolveFor('modernizr', 'icons', 'toaster', 'ngDialog', 'datatables','ngDeviceDetect', 'babylonjs', 'levenshtein'),
        controller: 'genCmdTaskCtrl'
      });

  } // routesConfig

})();

