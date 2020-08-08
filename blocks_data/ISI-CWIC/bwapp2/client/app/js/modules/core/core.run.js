(function() {
  'use strict';

  angular
      .module('app.core')
      .run(appRun);

  appRun.$inject = ['$rootScope', '$state', '$stateParams',  '$window', '$templateCache', 'Colors'];
  
  function appRun($rootScope, $state, $stateParams, $window, $templateCache, Colors) {

    // Set reference to access them from any scope
    $rootScope.$state = $state;
    $rootScope.$stateParams = $stateParams;
    $rootScope.$storage = $window.localStorage;

    // Uncomment this to disable template cache
    /*$rootScope.$on('$stateChangeStart', function(event, toState, toParams, fromState, fromParams) {
     if (typeof(toState) !== 'undefined'){
     $templateCache.remove(toState.templateUrl);
     }
     });*/

    // Allows to use branding color with interpolation
    // {{ colorByName('primary') }}
    $rootScope.colorByName = Colors.byName;

    // cancel click event easily
    $rootScope.cancel = function ($event) {
      $event.stopPropagation();
    };

    // Hooks Example
    // ----------------------------------- 
    $rootScope.isRole = function(usr, role){
      if(usr){
        if(role == 'guest'){ //there is always a default guest account via artwells:accounts-guest
          if(usr && usr.profile && usr.profile.guest) return usr.profile.guest;
        } else if(usr && usr.profile && usr.profile.roles)
          return (usr.profile.roles.indexOf(role) > -1);
      }
      return false;
    };

    // Hook not found
    $rootScope.$on('$stateNotFound',
      function (event, unfoundState/*, fromState, fromParams*/) {
        console.log(unfoundState.to); // "lazy.state"
        console.log(unfoundState.toParams); // {a:1, b:2}
        console.log(unfoundState.options); // {inherit:false} + default options
      });
    // Hook error
    $rootScope.$on('$stateChangeError',
      function (event, toState, toParams, fromState, fromParams, error) {
        // We can catch the error thrown when the $requireUser promise is rejected
        // and redirect the user back to the main page
        if (error === "AUTH_REQUIRED") {
          $state.go('main');
        }
        else if (error === "FORBIDDEN") {
          $state.go('app.root');
        }
        else console.warn(error);
      });
    // Hook success
    $rootScope.$on('$stateChangeSuccess',
      function (/*event, toState, toParams, fromState, fromParams*/) {
        // display new view from top
        $window.scrollTo(0, 0);
        // Save the route title
        $rootScope.currTitle = $state.current.title;
        $rootScope.dataloaded = false;
      });

    //setup account callbacks
    accountsUIBootstrap3.logoutCallback = function (err) {
      if (err) console.log("Error:" + err);
      Meteor.loginVisitor(); //force guest login upon logout
      $state.go('main');
    };
    Accounts.config({forbidClientAccountCreation: true});

    // Load a title dynamically
    $rootScope.currTitle = $state.current.title;
    $rootScope.pageTitle = function () {
      var title = $rootScope.app.name + ' - ' + ($rootScope.currTitle || $rootScope.app.description);
      document.title = title;
      return title;
    };

  }

})();

