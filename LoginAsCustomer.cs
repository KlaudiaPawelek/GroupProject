using System;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using NUnit.Framework;
using OpenQA.Selenium;
using OpenQA.Selenium.Firefox;
using OpenQA.Selenium.Support.UI;

namespace SeleniumTests
{
    [TestFixture]
    public class LoginAsCustomer
    {
        private IWebDriver driver;
        private StringBuilder verificationErrors;
        private string baseURL;
        private bool acceptNextAlert = true;
        
        [SetUp]
        public void SetupTest()
        {
            // Test can be run under Firefox (for example). It can be also other browser!
            driver = new FirefoxDriver();
            baseURL = "https://www.katalon.com/";
            verificationErrors = new StringBuilder();
        }
        
        // Catch exceptions and tell about problem, which occurs on web page.
        [TearDown]
        public void TeardownTest()
        {
            try
            {
                driver.Quit();
            }
            catch (Exception)
            {
                // Ignore errors if unable to close the browser
            }
            Assert.AreEqual("", verificationErrors.ToString());
        }
        
        // Method, which includes main automated test.
        // The aim of this test is to check, if login to page works correctly.
        // Additionally, test checks which elements are visible before login and after login for CUSTOMER account.
        [Test]
        public void TheLoginAsCustomerTest()
        {
            // Go to our web page.
            driver.Navigate().GoToUrl("http://ec2-35-176-2-229.eu-west-2.compute.amazonaws.com/drupal/");

            // Check if some elements, which are recognize by XPATH, are visible on page after open it, for anonymous user (user, which is not logged in yet).

            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Main navigation'])[1]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }
            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (!IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Add images and run'])[2]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }
            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (!IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Your samples'])[2]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }
            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (!IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Edit menu'])[3]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }
            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (!IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Add images and run'])[3]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }
            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (!IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Your samples'])[3]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }

            // Login to page

            driver.FindElement(By.LinkText("Log in")).Click();
            driver.FindElement(By.Id("edit-name")).Click();
            driver.FindElement(By.Id("edit-name")).Clear();
            driver.FindElement(By.Id("edit-name")).SendKeys("root");
            driver.FindElement(By.Id("edit-pass")).Click();
            driver.FindElement(By.Id("edit-pass")).Clear();
            driver.FindElement(By.Id("edit-pass")).SendKeys("droplets1994");
            driver.FindElement(By.Id("edit-submit")).Click();

            // Wait for element "HOME" after login to check, if after login everything is proper.
            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Home'])[2]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }

            // Check if some elements, which are recognize by XPATH, are visible on page after login as CUSTOMER.
            // For example: some elements in menu, which shall be visible only after login. If visible, click on it and go to this page.

            driver.FindElement(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Add images and run'])[2]/following::a[1]")).Click();
            driver.FindElement(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Your samples'])[2]/following::a[1]")).Click();
            driver.FindElement(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Edit menu'])[3]/following::a[1]")).Click();
            driver.FindElement(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Add images and run'])[3]/following::a[1]")).Click();
            driver.FindElement(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Your samples'])[3]/following::a[1]")).Click();


            // Logout and check, if some elements are hide correctly.

            driver.FindElement(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='My account'])[1]/following::a[1]")).Click();

            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Main navigation'])[1]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }
            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (!IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Add images and run'])[2]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }
            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (!IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Your samples'])[2]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }
            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (!IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Edit menu'])[3]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }
            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (!IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Add images and run'])[3]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }
            for (int second = 0;; second++) {
                if (second >= 60) Assert.Fail("timeout");
                try
                {
                    if (!IsElementPresent(By.XPath("(.//*[normalize-space(text()) and normalize-space(.)='Your samples'])[3]/following::a[1]"))) break;
                }
                catch (Exception)
                {}
                Thread.Sleep(1000);
            }

            //End of the test.

        }

        // Methods responsible for catching errors.
        // It means, if test passed or failed.
        
        private bool IsElementPresent(By by)
        {
            try
            {
                driver.FindElement(by);
                return true;
            }
            catch (NoSuchElementException)
            {
                return false;
            }
        }
        
        private bool IsAlertPresent()
        {
            try
            {
                driver.SwitchTo().Alert();
                return true;
            }
            catch (NoAlertPresentException)
            {
                return false;
            }
        }
        
        private string CloseAlertAndGetItsText() {
            try {
                IAlert alert = driver.SwitchTo().Alert();
                string alertText = alert.Text;
                if (acceptNextAlert) {
                    alert.Accept();
                } else {
                    alert.Dismiss();
                }
                return alertText;
            } finally {
                acceptNextAlert = true;
            }
        }
    }
}
